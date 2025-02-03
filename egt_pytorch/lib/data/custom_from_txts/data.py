import itertools
import os
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import SVDEncodingsGraphDataset
from ..graph_dataset import StructuralDataset


class CustomDataset(DatasetBase):
    def __init__(self, 
                 dataset_path,
                 dataset_name = '',
                 **kwargs
                 ):
        super().__init__(
                        dataset_name = dataset_name,
                         **kwargs,
                         )
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        data_idx, data_dict = read_graphs_from_txt(self.dataset_path,self.dataset_name)
        self.data_idx = data_idx
        self._dataset = data_dict
        
    @property
    def dataset(self):
        return self._dataset

    @property
    def record_tokens(self):
        try:
            return self._record_tokens
        except AttributeError:
            train_tokens = int(len(self._dataset)*0.8)
            validation_tokens = int(len(self._dataset)*0.9)
            self.train_idx = self.data_idx[:train_tokens]
            self.validation_idx = self.data_idx[train_tokens:validation_tokens]
            self.test_idx = self.data_idx[validation_tokens:]
            split = {'all':self.data_idx,
                    'training':self.train_idx, 
                    'validation':self.validation_idx, 
                    'test':self.test_idx}[self.split]
            self._record_tokens = split
            return self._record_tokens
    
    def read_record(self, token):
        graph = self._dataset[token]
        return graph


class CustomGraphDataset(GraphDataset,CustomDataset):
    pass

class CustomSVDGraphDataset(SVDEncodingsGraphDataset,CustomDataset):
    pass

class CustomStructuralGraphDataset(StructuralDataset,CustomGraphDataset):
    pass

class CustomStructuralSVDGraphDataset(StructuralDataset,CustomSVDGraphDataset):
    pass


# from utils.config import get_config
# cfg = get_config()
from GSN.utils import *
def add_GSN(data, id_type='path_graph'):
    args = {
            'id_type':id_type,
            # 'id_type':'cycle_graph',
            'induced':False,
            'edge_automorphism':'induced',
            'k':[7],
            'id_scope':'global',#'local'
            'custom_edge_list':None,'directed':False,
            'directed_orbits':False,'id_encoding':'one_hot_unique',
            'degree_encoding':'one_hot_unique','id_bins':None,
            'degree_bins':None,'id_strategy':'uniform','degree_strategy':'uniform',
            'id_range':None,'degree_range':None,'id_embedding':'one_hot_encoder',
            'd_out_id_embedding':None,'degree_embedding':'one_hot_encoder',
            'd_out_degree_embedding':None,'input_node_encoder':'None',
            'd_out_node_encoder':None,'edge_encoder':'None','d_out_edge_encoder':None,
            'multi_embedding_aggr':'sum','extend_dims':True,
            }
    args, count_fn, automorphism_fn = process_arguments(args)
    # id_scope = args['id_scope']
    id_type = args['id_type']
    k = args['k']
    subgraph_params = {'induced': args['induced'], 
                        'edge_list': args['custom_edge_list'],
                        'directed': args['directed'],
                        'directed_orbits': args['directed_orbits']}
    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    subgraph_dicts = []
    orbit_partition_sizes = []
    if 'edge_list' not in subgraph_params:
        raise ValueError('Edge list not provided.')
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
                                            automorphism_fn(edge_list=edge_list,
                                                            directed=subgraph_params['directed'],
                                                            directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph':subgraph, 'orbit_partition': orbit_partition, 
                                'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))
    new_data = GSN_prepare(data, subgraph_dicts, 
                subgraph_counts2ids,
                count_fn, orbit_partition_sizes, args)
    return new_data.degrees.cpu().detach().numpy().astype(np.int16), new_data.identifiers.cpu().detach().numpy().astype(np.int16)

def read_graphs_from_txt(raw_dir,name):
    path = os.path.join(raw_dir, f'{name}_node_attributes.txt')
    node_attrs_df = pd.read_csv(path, sep=',', header=None)
    node_attrs_df.index += 1
    path = os.path.join(raw_dir, f'{name}_A.txt')
    edge_index_df = pd.read_csv(path, sep=',', names=['source', 'target'])
    edge_index_df.index += 1
    path = os.path.join(raw_dir, f'{name}_edge_attributes.txt')
    edge_attrs_df = pd.read_csv(path, sep=',', header=None)
    edge_attrs_df.index += 1
    path = os.path.join(raw_dir, f'{name}_graph_indicator.txt')
    graph_idx_df = pd.read_csv(path, sep=',', names=['idx'])
    graph_idx_df.index += 1
    path = os.path.join(raw_dir, f'{name}_graph_labels.txt')
    # graph_labels_df = pd.read_csv(path, sep=',', names=['label'])
    graph_labels_df = pd.read_csv(path, sep=',', header=None)
    # graph_labels_df['label'] = graph_labels_df['label'] - 1
    graph_labels_df.index += 1

    data_dict = {}
    ids_list = graph_idx_df['idx'].unique()
    data_idx = []
    # co_mappings = [] #Canonical Ordering mapping
    for g_idx in tqdm(ids_list):
        data_idx.append(g_idx)
        graph = {}
        node_ids = graph_idx_df.loc[graph_idx_df['idx']==g_idx].index
        # Node features
        node_attributes = node_attrs_df.loc[node_ids, :]
        # Edges info
        edges = edge_index_df.loc[edge_index_df['source'].isin(node_ids)]
        edges_ids = edges.index
        # Edge features
        edge_attributes = edge_attrs_df.loc[edges_ids, :]
        # Graph label
        label = graph_labels_df.loc[g_idx]
        # Normalize the edges indices
        edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
        map_dict = {v.item():i for i,v in enumerate(torch.unique(edge_idx))}
        map_edge = torch.zeros_like(edge_idx)
        for k,v in map_dict.items():
            map_edge[edge_idx==k] = v
        # Convert the DataFrames into tensors 
        x = torch.tensor(node_attributes.to_numpy(), dtype=torch.float)
        edge_attr = torch.tensor(edge_attributes.to_numpy(), dtype=torch.float)
        edge_idx = map_edge.long()
        y = torch.tensor(label.to_numpy(), dtype=torch.long)
        # PyG graph
        graph_p = Data(x=x, edge_index=edge_idx, y=y, edge_attr=edge_attr)
        # NetworkX graph
        graph_x = to_networkx(graph_p, 
                            node_attrs=["x"], 
                            edge_attrs=["edge_attr"],
                            # graph_attrs=["y"], 
                            to_undirected=True, 
                            remove_self_loops=True)
        # Get Canonical Ordering, why: https://proceedings.neurips.cc/paper/2019/file/d0921d442ee91b896ad95059d13df618-Paper.pdf
        mapping = get_ordering(graph_x)
        # co_mappings.append(mapping)
        #reorder node features
        x_reordered = np.zeros_like(x.numpy())
        for i in range(x_reordered.shape[0]):
            x_reordered[i] = x.numpy()[list(mapping.keys())[i]]
        #reorder edge_idx
        temp = edge_idx.numpy()
        edge_idx_reordered = np.zeros_like(temp)
        for i, r in enumerate(temp):
            edge_idx_reordered[i] = [mapping[k] for k in r]
        # dictionary
        num_nodes = x.shape[0]
        graph['num_nodes'] = np.array(num_nodes,dtype='int16')
        graph['edges_old'] = edge_idx.numpy().T.astype(np.int16)
        graph['edges'] = edge_idx_reordered.T.astype(np.int16)
        graph['edge_features'] = edge_attr.numpy().astype(np.int16)
        graph['node_features_old'] = x.numpy().astype(np.float32)
        graph['node_features'] = x_reordered.astype(np.float32)
        graph['target'] = y.numpy().astype(np.int16)
        data_dict[g_idx] = graph
        
    data_idx = np.array(data_idx)
    return data_idx, data_dict#, co_mappings


# refer to https://colab.research.google.com/drive/1M--YX4dOSt3imDPdecPbjVX-T6Ae0_OG
def get_ordering(g):# a networkx graph
    node_k1 = dict(g.degree())  ## sort by degree
    node_k2 = nx.average_neighbor_degree(g)  ## sort by average neighbor degree
    node_closeness = nx.closeness_centrality(g)
    node_betweenness = nx.betweenness_centrality(g)
    node_sorting = list()
    for node_id in g.nodes:
        node_sorting.append((node_id, node_k1[node_id], node_k2[node_id], node_closeness[node_id], node_betweenness[node_id]))
        
    node_descending = sorted(node_sorting, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    mapping = dict()
    for i, node in enumerate(node_descending):
        mapping[node[0]] = i

    return mapping



