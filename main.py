import os
import random
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import itertools
from datetime import datetime
import pickle
from utils.utils import *
from utils.config import get_config
# model loader
from egt_pytorch.lib.models.egt_graphvae import EGT_GRAPHVAE
cfg = get_config()
if cfg.device == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cfg.device == 'cpu':
    device = 'cpu'

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed(seed = cfg.seed)


# dataloader
from egt_pytorch.lib.data.custom_from_txts import *
# split: 'all','training','validation','test'
train_loader = CustomStructuralSVDGraphDataset(dataset_path=f'./data/AAMG/{cfg.dataset}/',
                 dataset_name = cfg.dataset,split = 'all')
train_loader.cache()


def train(train_loader, model, _epoch, optimizer, scheduler):
    model.train()
    for i, graph in enumerate(train_loader):
        # prepare graph data
        graph['num_nodes'] = torch.from_numpy(graph['num_nodes']).unsqueeze(0).to(device)
        graph['target'] = torch.from_numpy(graph['target']).unsqueeze(0).to(device)
        graph['node_mask'] = torch.from_numpy(graph['node_mask']).unsqueeze(0).to(device)
        graph['svd_encodings'] = torch.from_numpy(graph['svd_encodings']).unsqueeze(0).to(device)
        graph['svd_encodings'] = 2*(torch.sigmoid(graph['svd_encodings'])) -1
        if graph['svd_encodings'].isnan().any() or not torch.all(torch.isfinite(graph['svd_encodings'])):
            continue
        graph['node_features'] = torch.from_numpy(graph['node_features']).unsqueeze(0).to(device)
        graph['distance_matrix'] = torch.from_numpy(graph['distance_matrix']).unsqueeze(0).to(device)
        graph['feature_matrix'] = torch.from_numpy(graph['feature_matrix']).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        
        if cfg.vaetype == 'vae' or cfg.vaetype == 'betavae':
            if not cfg.ifNED:
                adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                        z_g_tmp, z_g_mu_tmp, z_g_sigma_tmp, \
                        ws_A, ws_X, x_A, x_X = model(graph)
            if cfg.ifNED:
                adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                        z_g_n_graph, z_g_e_graph, z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                        ws_A, ws_X, x_A, x_X = model(graph)
            total_loss = 10 * adj_loss + 10 * adj_style_loss + 10 * fea_loss + 10 * fea_style_loss + kl_loss
            # print('adj_loss: ', adj_loss)
            # print('adj_style_loss: ', adj_style_loss)
            # print('fea_loss: ', fea_loss)
            # print('fea_style_loss: ', fea_style_loss)
            # print('kl_loss: ', kl_loss)
            
        if cfg.vaetype == 'vqvae':
            if not cfg.ifNED:
                adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, vq_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                        z_g_tmp, z_g_mu_tmp, z_g_sigma_tmp, z_quantized, emb_quantized, \
                        ws_A, ws_X, x_A, x_X = model(graph)
            if cfg.ifNED:
                adj_loss, adj_style_loss, fea_loss, fea_style_loss, kl_loss, vq_loss, \
                    Ag_pred, Ag_pred_style, Xg_pred, Xg_pred_style, \
                        z_n_quantized, emb_n_quantized, z_e_quantized, emb_e_quantized, z_g_n_graph, z_g_e_graph, \
                        z_g_mu, z_g_sigma, z_n_mu, z_n_sigma, z_e_mu, z_e_sigma, \
                        ws_A, ws_X, x_A, x_X = model(graph)
            total_loss = 10 * adj_loss + 10 * adj_style_loss + 10 * fea_loss + 10 * fea_style_loss + kl_loss + vq_loss
            # print('adj_loss: ', adj_loss)
            # print('adj_style_loss: ', adj_style_loss)
            # print('fea_loss: ', fea_loss)
            # print('fea_style_loss: ', fea_style_loss)
            # print('kl_loss: ', kl_loss)
            # print('vq_loss: ', vq_loss)

        #LOSS BACKPROP
        total_loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            print(f"Factor = {i}, Learning Rate = {optimizer.param_groups[0]['lr']}")
            scheduler.step()

        #PRINT and save LOSSES
        print(f'Epoch [{_epoch:03d}/{cfg.num_epochs:03d}], Step [{i:04d}/{len(train_loader):04d}], lr {optimizer.param_groups[0]["lr"]}, \
              total_loss {str(total_loss)}'
                + "\n")

        if cfg.vaetype == 'vae' or cfg.vaetype == 'betavae':
            with open(os.path.join(cfg.txt_path, f"loss.txt"),"a",encoding="utf-8") as file:
                file.write(
                    f'{datetime.now()} Epoch [{_epoch:03d}/{cfg.num_epochs:03d}], \
                        Step [{i:04d}/{len(train_loader):04d}], lr {optimizer.param_groups[0]["lr"]}, \
                        adj_loss {adj_loss}, adj_style_loss {adj_style_loss}, fea_loss {fea_loss}, \
                        fea_style_loss {fea_style_loss}, kl_loss {kl_loss}'
                    + "\n")
        if cfg.vaetype == 'vqvae':
            with open(os.path.join(cfg.txt_path, f"loss.txt"),"a",encoding="utf-8") as file:
                file.write(
                    f'{datetime.now()} Epoch [{_epoch:03d}/{cfg.num_epochs:03d}], \
                        Step [{i:04d}/{len(train_loader):04d}], lr {optimizer.param_groups[0]["lr"]}, \
                        adj_loss {adj_loss}, adj_style_loss {adj_style_loss}, fea_loss {fea_loss}, \
                        fea_style_loss {fea_style_loss}, kl_loss {kl_loss}, vq_loss {vq_loss}'
                    + "\n")
                
        break
            
    #SAVE CHECKPOINT
    if _epoch % 1 == 0:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': _epoch,
            }, cfg.model_path + 'FP_VAE' + '_%d' % _epoch + '.pth')

def validation(model, _epoch):
    model.eval()
    data_gen_dict = {}
    for i in range(cfg.num_gen_samples):
        graph = {}
        if not cfg.ifNED:
            z_rand = torch.randn(1, cfg.ZDIM).to(device)
            A,X,\
                z_quantized, emb_quantized, \
                Ag_quantized, Xg_quantized, \
                ws_A, x_A, Ag_pred_style, \
                ws_X, x_X, Xg_pred_style \
                = model.inference(z_rand)
            graph['z_quantized'] = z_quantized
            graph['emb_quantized'] = emb_quantized
            graph['A_quantized'] = Ag_quantized
            graph['X_quantized'] = Xg_quantized
        if cfg.ifNED:
            z_n_rand = torch.randn(1, cfg.ZDIM).to(device)
            z_e_rand = torch.randn(1, cfg.ZDIM).to(device)
            A,X,\
                z_n_quantized, emb_n_quantized, \
                z_e_quantized, emb_e_quantized, \
                Ag_quantized, Xg_quantized, \
                ws_A, x_A, Ag_pred_style, \
                ws_X, x_X, Xg_pred_style \
                = model.inference_NED(z_n_rand, z_e_rand)
            graph['z_n_quantized'] = z_n_quantized
            graph['emb_n_quantized'] = emb_n_quantized
            graph['z_e_quantized'] = z_e_quantized
            graph['emb_e_quantized'] = emb_e_quantized
            graph['A_quantized'] = Ag_quantized
            graph['X_quantized'] = Xg_quantized
        graph['A'] = A
        graph['X'] = X
        graph['ws_A'] = ws_A
        graph['x_A'] = x_A
        graph['A_style'] = Ag_pred_style
        graph['ws_X'] = ws_X
        graph['x_X'] = x_X
        graph['X_style'] = Xg_pred_style
        data_gen_dict[i] = graph
    with open(f'{cfg.sample_path}/sample_{_epoch}.pickle', 'wb') as handle:
        pickle.dump(data_gen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    model = EGT_GRAPHVAE(max_num_nodes = train_loader.max_nodes.item())
    print(model)
    # Initializing the weights with the normal initialization method 
    for param in model.parameters():
        torch.nn.init.normal_(param, 
                            mean=0, std=1) 

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.txt_path, exist_ok=True)
    os.makedirs(cfg.sample_path, exist_ok=True)
    _epoch, model, optimizer, scheduler = load_model(cfg.model_path, model=model)

    while _epoch < cfg.num_epochs:
        train(train_loader, model, _epoch, optimizer, scheduler)
        if _epoch % cfg.save_samples == 0:
            print(f"Generating {cfg.num_gen_samples} samples for Epoch {_epoch}")
            validation(model, _epoch)
        _epoch += 1


if __name__ == '__main__':
    main()

