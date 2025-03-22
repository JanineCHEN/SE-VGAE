import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .egt import EGT
from .egt_layers import VirtualNodes
from .styleDecoder import Generator_2D as StyleDecoder_2D
from .styleDecoder import Generator_1D as StyleDecoder_1D
from .nearest_embed import NearestEmbed, NearestEmbedEMA
from .style_loss import StyleGAN2Loss
from utils.config import get_config

cfg = get_config()
if cfg.device == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cfg.device == 'cpu':
    device = 'cpu'


ZDIM = cfg.ZDIM
# NODE_FEATURES_OFFSET = 128
if 'basic' in cfg.dataset:
    if '25' in cfg.dataset:
        NUM_NODE_FEATURES = 22
    if '6' in cfg.dataset:
        NUM_NODE_FEATURES = 6
if 'full' in cfg.dataset:
    if '25' in cfg.dataset:
        NUM_NODE_FEATURES = 1022
    if '6' in cfg.dataset:
        NUM_NODE_FEATURES = 1006
# EDGE_FEATURES_OFFSET = 8
if '25' in cfg.dataset:
    NUM_EDGE_FEATURES = 6 # remember to delete cache to make the update work!!!!!!
if '6' in cfg.dataset:
    NUM_EDGE_FEATURES = 4 # remember to delete cache to make the update work!!!!!!
VAE = False
BETA_VAE = False
VQ_VAE = False
if cfg.vaetype == 'vae':
    VAE = True
if cfg.vaetype == 'betavae':
    BETA_VAE = True
if cfg.vaetype == 'vqvae':
    VQ_VAE = True
VQ_EMA = False
ifGAN = False

class GINLayer(torch.nn.Module):
    def __init__(self, num_feature, eps, batch_size, max_num_nodes, edge_width):
        super().__init__()
        self.num_feature = num_feature
        self.eps = eps
        self.batch_size = batch_size
        self.max_num_nodes = max_num_nodes
        self.edge_width = edge_width
        
        self.MLP_GIN = nn.Sequential(
            nn.Linear(self.num_feature, self.num_feature),
            nn.PReLU(),
            nn.LayerNorm(self.num_feature)
            )
        self.edge_feature_mapping = nn.Linear(self.edge_width, 1)
        
    def forward(self, A, X):
        X_tmp_1 = (1+self.eps)*X
        A = self.edge_feature_mapping(A)
        A = A.view(self.batch_size, self.max_num_nodes, -1)
        X_tmp_2 = torch.matmul(A, X)
        X_tmp = X_tmp_1 + X_tmp_2
        X_new = self.MLP_GIN(X_tmp)
        return X_new
    
class Layer_edge(torch.nn.Module):
    def __init__(self, dim_zg, batch_size, max_num_nodes, edge_width):
        super().__init__()
        self.dim_zg = dim_zg
        self.batch_size = batch_size
        self.max_num_nodes = max_num_nodes
        self.edge_width = edge_width
        self.edge_feature_mapping = nn.Linear(self.edge_width, 1)
        self.feature_mapping = nn.Flatten()
        self.z_mapping = nn.Linear(self.max_num_nodes*self.max_num_nodes, self.dim_zg)
        
    def forward(self, A):
        A = self.edge_feature_mapping(A)
        A = A.view(self.batch_size, self.max_num_nodes, -1)
        A = self.feature_mapping(A)
        A = self.z_mapping(A)
        return A


class vector_quantizer(nn.Module):
    """Vector Quantizer"""

    def __init__(self, hidden=ZDIM, k=ZDIM, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(vector_quantizer, self).__init__()

        self.emb_size = k
        if not VQ_EMA:
            self.emb = NearestEmbed(k, self.emb_size)
        if VQ_EMA:
            self.emb = NearestEmbedEMA(k, self.emb_size)

        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.vq_loss = 0
        self.commit_loss = 0

    def forward(self, z_g):
        z_q, _ = self.emb(z_g, weight_sg=True)
        emb, _ = self.emb(z_g.detach())

        return z_q, emb

    def loss_function(self, z_g, emb):
        self.vq_loss = F.mse_loss(emb, z_g.detach())
        self.commit_loss = F.mse_loss(z_g, emb.detach())

        return self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss


class EGT_GRAPHVAE(EGT):
    def __init__(self,
                 max_num_nodes, #Must be a power of 2
                 batch_size        = 1,
                 upto_hop          = 16,
                 mlp_ratios        = [1., 1.],
                 num_virtual_nodes = 0,
                 svd_encodings     = cfg.svdDIM,
                 output_dim        = 7, #num of node class
                 dim_zg            = ZDIM,
                 num_layer_vae     = 1, #GINs for mapping transformer outputs
                 **kwargs):
        super().__init__(node_ended=True, **kwargs)
        self.max_num_nodes     = max_num_nodes
        self.batch             = batch_size,
        self.upto_hop          = upto_hop
        self.mlp_ratios        = mlp_ratios
        self.num_virtual_nodes = num_virtual_nodes
        self.svd_encodings     = svd_encodings
        self.output_dim        = output_dim
        self.dim_zg            = dim_zg
        self.num_layer_w = num_layer_vae
        self.vector_quantizer = vector_quantizer()
        self.vector_quantizer_n = vector_quantizer()
        self.vector_quantizer_e = vector_quantizer()

        # make sure self.max_num_nodes is a power of 2
        if not math.log2(self.max_num_nodes).is_integer():
            i = 1
            while i < self.max_num_nodes: i *= 2
            self.max_num_nodes = i

        ## Transformer Encoder 
        self.nodef_embed = nn.Linear(NUM_NODE_FEATURES, self.node_width)
        if self.svd_encodings:
            self.svd_embed = nn.Linear(self.svd_encodings*2, self.node_width)
        
        self.dist_embed = nn.Embedding(self.upto_hop+2, self.edge_width)
        self.featm_embed = nn.Embedding(NUM_EDGE_FEATURES+1,
                                        self.edge_width, #the size of each embedding vector 
                                        )
        
        if self.num_virtual_nodes > 0:
            self.vn_layer = VirtualNodes(self.node_width, self.edge_width, 
                                         self.num_virtual_nodes)
        
        self.final_ln_h = nn.LayerNorm(self.node_width)
        self.final_ln_e = nn.LayerNorm(self.edge_width)
        mlp_dims = [self.node_width * max(self.num_virtual_nodes, 1)]\
                    +[round(self.node_width*r) for r in self.mlp_ratios]\
                        +[self.output_dim]
        self.mlp_layers = nn.ModuleList([nn.Linear(mlp_dims[i],mlp_dims[i+1])
                                         for i in range(len(mlp_dims)-1)])
        self.mlp_fn = getattr(F, self.activation)

        # GCN Encoder     
        ## GIN for global filters
        self.eps_g = nn.Parameter(torch.zeros(self.num_layer_w))
        self.gin_g = torch.nn.ModuleList()
        for i in range(self.num_layer_w):
            self.gin_g.append(GINLayer(self.node_width, self.eps_g[i], self.batch[0], self.max_num_nodes, self.edge_width))

        ## Edge mapper
        self.layer_e = torch.nn.ModuleList()
        for i in range(self.num_layer_w):
            self.layer_e.append(Layer_edge(self.dim_zg, self.batch[0], self.max_num_nodes, self.edge_width))

        ## Compute mu and sigma for VAE
        self.mu_g = nn.Sequential(
            nn.Linear(self.node_width, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            )
        self.sigma_g = nn.Sequential(
            nn.Linear(self.node_width, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            )
        self.mu_g_e = nn.Sequential(
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            )
        self.sigma_g_e = nn.Sequential(
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            )

        ## Decoder
        # Global
        self.GlobalPred_A = nn.Sequential(
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            nn.Linear(self.dim_zg, (self.max_num_nodes**2)*NUM_EDGE_FEATURES),
            nn.PReLU(),
            nn.LayerNorm((self.max_num_nodes**2)*NUM_EDGE_FEATURES),
            )
        self.GlobalPred_Attr = nn.Sequential(
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.PReLU(),
            nn.LayerNorm(self.dim_zg),
            nn.Linear(self.dim_zg, self.max_num_nodes*NUM_NODE_FEATURES),
            nn.PReLU(),
            nn.LayerNorm(self.max_num_nodes*NUM_NODE_FEATURES),
            )
        # style-based decoder
        self.style_decoder_2D = StyleDecoder_2D(z_dim=self.dim_zg,         # Input latent (Z) dimensionality.
                                        c_dim=0,              # # Conditioning label (C) dimensionality, 0 = no label.
                                        w_dim=ZDIM,            # Intermediate latent (W) dimensionality.
                                        img_resolution=self.max_num_nodes,# num of edge dimension. Must be a multiple of 32
                                        img_channels=NUM_EDGE_FEATURES,# edge feature channels
                                        )
        self.style_decoder_1D = StyleDecoder_1D(z_dim=self.dim_zg,         # Input latent (Z) dimensionality.
                                        c_dim=0,              # # Conditioning label (C) dimensionality, 0 = no label.
                                        w_dim=ZDIM,            # Intermediate latent (W) dimensionality.
                                        img_resolution=self.max_num_nodes,# num of edge dimension. Must be a multiple of 32
                                        img_channels=NUM_NODE_FEATURES,# edge feature channels
                                        )
        self.prelu = nn.PReLU()
        self.styleloss = StyleGAN2Loss(device)
        self.z_mapping = nn.Linear(self.dim_zg*2, self.dim_zg)

    # Convert original input graph dictionary
    def input_block(self, inputs):
        g = super().input_block(inputs)
        nodef = g.node_features#.long()              # (b,i,f)
        nodem = g.node_mask.float()                 # (b,i)
        dm0 = g.distance_matrix                     # (b,i,j)
        dm = dm0.long().clamp(max=self.upto_hop+1)  # (b,i,j)
        featm = g.feature_matrix.long()             # (b,i,j,f)
        h = self.nodef_embed(nodef)
        if self.svd_encodings:
            svd = self.svd_embed(g.svd_encodings)
            h = h + svd
        e = self.dist_embed(dm)\
              + self.featm_embed(featm).sum(dim=3)  # (b,i,j,f,e) -> (b,i,j,e)

        g.mask = (nodem[:,:,None,None] * nodem[:,None,:,None] - 1)*1e9
        g.h, g.e = h, e
        return g
    
    def final_embedding(self, g):
        # for layernorm before decoder initialization
        h, e = g.h, g.e
        h = self.final_ln_h(h)
        e = self.final_ln_e(e)
        g.h, g.e = h, e
        return g
    
    # not needed for vae
    def output_block(self, g):
        h = g.h
        h = self.mlp_layers[0](h)
        # for layer in self.mlp_layers[1:]:
        for layer in self.mlp_layers[1:-1]:
            h = layer(self.mlp_fn(h))
        g.h = h
        return g.h, g.e

    # Encoder process
    def encoder(self, A, X):
        if not cfg.ifNED:
            for layer in self.gin_g:
                X = layer(A, X)
            z_g = X 
            if cfg.gintype == 'mean':
                z_g = torch.mean(z_g, dim = 1)
            if cfg.gintype == 'sum':
                z_g = torch.sum(z_g, dim = 1)
            z_g_mu = self.mu_g(z_g)
            z_g_mu = z_g_mu.to(device)
            z_g_sigma = self.sigma_g(z_g)
            z_g_sigma = z_g_sigma.to(device)
            z_g_graph = z_g_mu + torch.randn(z_g_sigma.size()).to(device) * torch.exp(z_g_sigma)
            return z_g_graph, z_g_mu, z_g_sigma
        if cfg.ifNED:
            for layer in self.gin_g:
                X = layer(A, X)
            z_g = X 
            if cfg.gintype == 'mean':
                z_g = torch.mean(z_g, dim = 1)
            if cfg.gintype == 'sum':
                z_g = torch.sum(z_g, dim = 1)
            z_g_mu = self.mu_g(z_g)
            z_g_mu = z_g_mu.to(device)
            z_g_sigma = self.sigma_g(z_g)
            z_g_sigma = z_g_sigma.to(device)
            z_g_graph = z_g_mu + torch.randn(z_g_sigma.size()).to(device) * torch.exp(z_g_sigma)
            if cfg.gintype == 'mean':
                z_X = torch.mean(X, dim = 1)
            if cfg.gintype == 'sum':
                z_X = torch.sum(X, dim = 1)
            z_n_mu = self.mu_g(z_X)
            z_n_mu = z_n_mu.to(device)
            z_n_sigma = self.sigma_g(z_X)
            z_n_sigma = z_n_sigma.to(device)
            z_n_graph = z_n_mu + torch.randn(z_n_sigma.size()).to(device) * torch.exp(z_n_sigma)
            z_g_n_graph = torch.cat((z_n_graph, z_g_graph), 1)
            z_g_n_graph = self.z_mapping(z_g_n_graph)
            for layer in self.layer_e:
                A = layer(A)
            z_e_mu = self.mu_g_e(A)
            z_e_mu = z_e_mu.to(device)
            z_e_sigma = self.sigma_g_e(A)
            z_e_sigma = z_e_sigma.to(device)
            z_e_graph = z_e_mu + torch.randn(z_e_sigma.size()).to(device) * torch.exp(z_e_sigma)
            z_g_e_graph = torch.cat((z_e_graph, z_g_graph), 1)
            z_g_e_graph = self.z_mapping(z_g_e_graph)
            return z_g_n_graph, z_g_e_graph, z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma
        
    # Decoder process
    def decoder(self, z_g):
        # adjacency decoder
        Ag = self.GlobalPred_A(z_g).view(self.batch[0], self.max_num_nodes, self.max_num_nodes, NUM_EDGE_FEATURES)
        Ag = torch.sigmoid(Ag)
        
        # attribute decoder
        Xg = self.GlobalPred_Attr(z_g).view(self.batch[0], self.max_num_nodes, NUM_NODE_FEATURES)
        Xg = torch.sigmoid(Xg)
        return Ag, Xg
    
    def decoder_NED(self, z_n, z_e):
        # adjacency decoder
        Ag = self.GlobalPred_A(z_e).view(self.batch[0], self.max_num_nodes, self.max_num_nodes, NUM_EDGE_FEATURES)
        Ag = torch.sigmoid(Ag)
        # attribute decoder
        Xg = self.GlobalPred_Attr(z_n).view(self.batch[0], self.max_num_nodes, NUM_NODE_FEATURES)
        Xg = torch.sigmoid(Xg)
        return Ag, Xg


    # Combine encoder and decoder process
    def vae(self, A_pad, X_pad):
        # encoder
        z_g, z_g_mu, z_g_sigma = self.encoder(A_pad, X_pad)
        # adjacency decoder & attribute decoder
        Ag_pred, Xg_pred = self.decoder(z_g)
        # style-based adjacency decoder
        ws_A, x_A, Ag_pred_style = self.style_decoder_2D(z=z_g,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
        Ag_pred_style = Ag_pred_style.permute((0, 2, 3, 1))
        x_A = x_A.permute((0, 2, 3, 1)) # original adjacency matrix/edge embedding features
        Ag_pred_style = torch.sigmoid(Ag_pred_style)
        # style-based attribute decoder
        ws_X, x_X, Xg_pred_style = self.style_decoder_1D(z=z_g,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
        Xg_pred_style = Xg_pred_style.permute((0, 2, 1))
        x_X = x_X.permute((0, 2, 1)) # original node attribute embedding features
        Xg_pred_style = torch.sigmoid(Xg_pred_style)
        return z_g, z_g_mu, z_g_sigma, ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style
    
    def reparametrize(self, mu, sigma):
        std = sigma.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def beta_vae(self, A_pad, X_pad):
        if not cfg.ifNED:
            # encoder
            _, z_g_mu, z_g_sigma = self.encoder(A_pad, X_pad)
            z_g = self.reparametrize(z_g_mu, z_g_sigma)
            # adjacency decoder & attribute decoder
            Ag_pred, Xg_pred = self.decoder(z_g)
            # style-based adjacency decoder
            ws_A, x_A, Ag_pred_style = self.style_decoder_2D(z=z_g,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Ag_pred_style = Ag_pred_style.permute((0, 2, 3, 1))
            x_A = x_A.permute((0, 2, 3, 1)) # original adjacency matrix/edge embedding features
            Ag_pred_style = torch.sigmoid(Ag_pred_style)
            # style-based attribute decoder
            ws_X, x_X, Xg_pred_style = self.style_decoder_1D(z=z_g,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Xg_pred_style = Xg_pred_style.permute((0, 2, 1))
            x_X = x_X.permute((0, 2, 1)) # original node attribute embedding features
            Xg_pred_style = torch.sigmoid(Xg_pred_style)
            return z_g, z_g_mu, z_g_sigma, ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style
        if cfg.ifNED:
            z_g_n_graph, z_g_e_graph, z_g_mu, z_g_sigma, \
                z_n_mu, z_n_sigma, z_e_mu, z_e_sigma = self.encoder(A_pad, X_pad)
            Ag_pred, Xg_pred = self.decoder_NED(z_g_n_graph, z_g_e_graph)
            # style-based adjacency decoder
            ws_A, x_A, Ag_pred_style = self.style_decoder_2D(z=z_g_e_graph,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Ag_pred_style = Ag_pred_style.permute((0, 2, 3, 1))
            x_A = x_A.permute((0, 2, 3, 1)) # original adjacency matrix/edge embedding features
            Ag_pred_style = torch.sigmoid(Ag_pred_style)
            # style-based attribute decoder
            ws_X, x_X, Xg_pred_style = self.style_decoder_1D(z=z_g_n_graph,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Xg_pred_style = Xg_pred_style.permute((0, 2, 1))
            x_X = x_X.permute((0, 2, 1)) # original node attribute embedding features
            Xg_pred_style = torch.sigmoid(Xg_pred_style)
            return z_g_n_graph, z_g_e_graph, z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style
            
    
    def vq_vae(self, A_pad, X_pad):
        if not cfg.ifNED:
            # encoder
            z_g, z_g_mu, z_g_sigma = self.encoder(A_pad, X_pad)
            # A VectorQuantizer class which transform the encoder output 
            # into a discrete one-hot vector that is the index of the closest embedding vector
            z_quantized, emb_quantized = self.vector_quantizer(z_g)
            # adjacency decoder & attribute decoder
            Ag_pred, Xg_pred = self.decoder(z_quantized)
            # style-based adjacency decoder
            ws_A, x_A, Ag_pred_style = self.style_decoder_2D(z=z_quantized,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Ag_pred_style = Ag_pred_style.permute((0, 2, 3, 1))
            x_A = x_A.permute((0, 2, 3, 1)) # original adjacency matrix/edge embedding features
            Ag_pred_style = torch.sigmoid(Ag_pred_style)
            # style-based attribute decoder
            ws_X, x_X, Xg_pred_style = self.style_decoder_1D(z=z_quantized,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Xg_pred_style = Xg_pred_style.permute((0, 2, 1))
            x_X = x_X.permute((0, 2, 1)) # original node attribute embedding features
            Xg_pred_style = torch.sigmoid(Xg_pred_style)
            return z_quantized, emb_quantized, z_g, z_g_mu, z_g_sigma, ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style

        if cfg.ifNED:
            # encoder
            z_g_n_graph, z_g_e_graph, z_g_mu, z_g_sigma, \
                z_n_mu, z_n_sigma, z_e_mu, z_e_sigma = self.encoder(A_pad, X_pad)
            # A VectorQuantizer class which transform the encoder output 
            # into a discrete one-hot vector that is the index of the closest embedding vector
            z_n_quantized, emb_n_quantized = self.vector_quantizer_n(z_g_n_graph)
            z_e_quantized, emb_e_quantized = self.vector_quantizer_e(z_g_e_graph)
            # adjacency decoder & attribute decoder
            Ag_pred, Xg_pred = self.decoder_NED(z_n_quantized, z_e_quantized)
            # style-based adjacency decoder
            ws_A, x_A, Ag_pred_style = self.style_decoder_2D(z=z_e_quantized,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Ag_pred_style = Ag_pred_style.permute((0, 2, 3, 1))
            x_A = x_A.permute((0, 2, 3, 1)) # original adjacency matrix/edge embedding features
            Ag_pred_style = torch.sigmoid(Ag_pred_style)
            # style-based attribute decoder
            ws_X, x_X, Xg_pred_style = self.style_decoder_1D(z=z_n_quantized,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
            Xg_pred_style = Xg_pred_style.permute((0, 2, 1))
            x_X = x_X.permute((0, 2, 1)) # original node attribute embedding features
            Xg_pred_style = torch.sigmoid(Xg_pred_style)
            return z_n_quantized, emb_n_quantized, z_e_quantized, emb_e_quantized, z_g_n_graph, z_g_e_graph, \
                z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style

 
    def pad_data(self, a, attr):
        max_a = torch.zeros(self.batch[0], self.max_num_nodes, self.max_num_nodes, a.shape[3])
        max_a[:, :a.shape[1], :a.shape[2], :] = a
        max_attr = torch.zeros(self.batch[0], self.max_num_nodes, attr.shape[2])
        max_attr[:, :attr.shape[1], :attr.shape[2]] = attr
        return max_a, max_attr

    def forward(self, inputs):
        A_input = inputs['feature_matrix'].detach().clone()
        X_input = inputs['node_features'].detach().clone()
        A_input_pad, X_input_pad = self.pad_data(A_input, X_input)
        A_input_pad = A_input_pad.to(device)
        X_input_pad = X_input_pad.to(device)
        # get input format ready for transformer
        g = self.input_block(inputs)
        ## Transformer Encoder
        for layer in self.EGT_layers:
            g = layer(g)
            g.h = self.prelu(g.h)
            g.e = self.prelu(g.e)
            g = self.final_embedding(g) # for layernorm

        ## VAE
        A_pad, X_pad = self.pad_data(g.e, g.h)
        A_pad = A_pad.to(device)
        X_pad = X_pad.to(device)

        if BETA_VAE:
            if not cfg.ifNED:
                z_g_tmp, z_g_mu_tmp, z_g_sigma_tmp, \
                    ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style = self.beta_vae(A_pad, X_pad)
                kl_loss = torch.mean(-(0.5) * (1 + z_g_sigma_tmp - z_g_mu_tmp**2 - torch.exp(z_g_sigma_tmp) ** 2))
            if cfg.ifNED:
                z_g_n_graph, z_g_e_graph, z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                    ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style = self.beta_vae(A_pad, X_pad)
                kl_loss = torch.mean(-(0.5) * (1 + z_g_sigma - z_g_mu**2 - torch.exp(z_g_sigma) ** 2)) + \
                            torch.mean(-(0.5) * (1 + z_n_sigma - z_n_mu**2 - torch.exp(z_n_sigma) ** 2)) + \
                            torch.mean(-(0.5) * (1 + z_e_sigma - z_e_mu**2 - torch.exp(z_e_sigma) ** 2))
        if VQ_VAE:
            if not cfg.ifNED:
                z_quantized, emb_quantized, z_g_tmp, z_g_mu_tmp, z_g_sigma_tmp, \
                    ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style = self.vq_vae(A_pad, X_pad)
                vq_loss = self.vector_quantizer.loss_function(z_g_tmp, emb_quantized)
                kl_loss = torch.mean(-(0.5) * (1 + z_g_sigma_tmp - z_g_mu_tmp**2 - torch.exp(z_g_sigma_tmp) ** 2))
            if cfg.ifNED:
                z_n_quantized, emb_n_quantized, z_e_quantized, emb_e_quantized, z_g_n_graph, z_g_e_graph, \
                    z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                    ws_A, ws_X, x_A, x_X, Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style = self.vq_vae(A_pad, X_pad)
                vq_loss = self.vector_quantizer_n.loss_function(z_g_n_graph, emb_n_quantized) + \
                    self.vector_quantizer_e.loss_function(z_g_e_graph, emb_e_quantized)
                kl_loss = torch.mean(-(0.5) * (1 + z_g_sigma - z_g_mu**2 - torch.exp(z_g_sigma) ** 2)) + \
                            torch.mean(-(0.5) * (1 + z_n_sigma - z_n_mu**2 - torch.exp(z_n_sigma) ** 2)) + \
                            torch.mean(-(0.5) * (1 + z_e_sigma - z_e_mu**2 - torch.exp(z_e_sigma) ** 2))
                
        ## LOSS
        adj_loss = F.binary_cross_entropy(Ag_pred, A_input_pad)
        adj_style_loss = F.binary_cross_entropy(Ag_pred_style, A_input_pad)
        fea_loss = F.binary_cross_entropy(Xg_pred, X_input_pad)
        fea_style_loss = F.binary_cross_entropy(Xg_pred_style, X_input_pad)
        
        if BETA_VAE:
            if not cfg.ifNED:
                return adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                    z_g_tmp, z_g_mu_tmp, z_g_sigma_tmp, \
                    ws_A, ws_X, x_A, x_X
            if cfg.ifNED:
                return adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                    z_g_n_graph, z_g_e_graph, z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                    ws_A, ws_X, x_A, x_X
        if VQ_VAE:
            if not cfg.ifNED:
                return adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, vq_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                    z_g_tmp, z_g_mu_tmp, z_g_sigma_tmp, z_quantized, emb_quantized, \
                    ws_A, ws_X, x_A, x_X
            if cfg.ifNED:
                return adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, vq_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                    z_n_quantized, emb_n_quantized, z_e_quantized, emb_e_quantized, z_g_n_graph, z_g_e_graph, \
                    z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                    ws_A, ws_X, x_A, x_X

    def inference(self, z_g):
        A,X = self.decoder(z_g)
        z_quantized, emb_quantized = self.vector_quantizer(z_g)
        # adjacency decoder & attribute decoder
        Ag_quantized, Xg_quantized = self.decoder(z_quantized)
        # style-based adjacency decoder
        ws_A, x_A, Ag_pred_style = self.style_decoder_2D(z=z_g,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
        Ag_pred_style = Ag_pred_style.permute((0, 2, 3, 1))
        x_A = x_A.permute((0, 2, 3, 1)) # original adjacency matrix/edge embedding features
        Ag_pred_style = torch.sigmoid(Ag_pred_style)
        # style-based attribute decoder
        ws_X, x_X, Xg_pred_style = self.style_decoder_1D(z=z_g,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
        Xg_pred_style = Xg_pred_style.permute((0, 2, 1))
        x_X = x_X.permute((0, 2, 1)) # original node attribute embedding features
        Xg_pred_style = torch.sigmoid(Xg_pred_style)
        return A.detach().cpu().numpy(),X.detach().cpu().numpy(),\
                z_quantized.detach().cpu().numpy(), emb_quantized.detach().cpu().numpy(), \
                Ag_quantized.detach().cpu().numpy(), Xg_quantized.detach().cpu().numpy(), \
                ws_A.detach().cpu().numpy(), x_A.detach().cpu().numpy(), Ag_pred_style.detach().cpu().numpy(), \
                ws_X.detach().cpu().numpy(), x_X.detach().cpu().numpy(), Xg_pred_style.detach().cpu().numpy()
    
    def inference_NED(self, z_n, z_e):
        A,X = self.decoder_NED(z_n, z_e)
        z_n_quantized, emb_n_quantized = self.vector_quantizer_n(z_n)
        z_e_quantized, emb_e_quantized = self.vector_quantizer_e(z_e)
        # adjacency decoder & attribute decoder
        Ag_quantized, Xg_quantized = self.decoder_NED(z_n_quantized, z_e_quantized)
        # style-based adjacency decoder
        ws_A, x_A, Ag_pred_style = self.style_decoder_2D(z=z_e_quantized,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
        Ag_pred_style = Ag_pred_style.permute((0, 2, 3, 1))
        x_A = x_A.permute((0, 2, 3, 1)) # original adjacency matrix/edge embedding features
        Ag_pred_style = torch.sigmoid(Ag_pred_style)
        # style-based attribute decoder
        ws_X, x_X, Xg_pred_style = self.style_decoder_1D(z=z_n_quantized,c= torch.zeros(1)) # Ag_pred_style: B, NUM_EDGE_FEATURES, N_nodes, N_nodes
        Xg_pred_style = Xg_pred_style.permute((0, 2, 1))
        x_X = x_X.permute((0, 2, 1)) # original node attribute embedding features
        Xg_pred_style = torch.sigmoid(Xg_pred_style)
        return A.detach().cpu().numpy(),X.detach().cpu().numpy(),\
                z_n_quantized.detach().cpu().numpy(), emb_n_quantized.detach().cpu().numpy(), \
                z_e_quantized.detach().cpu().numpy(), emb_e_quantized.detach().cpu().numpy(), \
                Ag_quantized.detach().cpu().numpy(), Xg_quantized.detach().cpu().numpy(), \
                ws_A.detach().cpu().numpy(), x_A.detach().cpu().numpy(), Ag_pred_style.detach().cpu().numpy(), \
                ws_X.detach().cpu().numpy(), x_X.detach().cpu().numpy(), Xg_pred_style.detach().cpu().numpy()
    
