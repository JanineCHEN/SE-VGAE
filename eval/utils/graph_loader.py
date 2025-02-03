import networkx as nx
import torch
import os
import pickle
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.convert import from_networkx

def load_graphs(name='ENZYMES', raw_dir = './data/'):
    '''
    load many graphs, e.g. enzymes
    :return: a list of networkx graphs
    '''
    # Read the files' content as Pandas DataFrame. Nodes and graphs ids
    # are based on the file row-index, we adjust the DataFrames indices
    # by starting from 1 instead of 0.
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
    graph_labels_df = pd.read_csv(path, sep=',', names=['label'])
    graph_labels_df['label'] = graph_labels_df['label'] - 1
    graph_labels_df.index += 1
    
    PyG_graph_list = []
    networkX_graph_list = []
    ids_list = graph_idx_df['idx'].unique()
    for g_idx in ids_list:
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
        graph = Data(x=x, edge_index=edge_idx, y=y, edge_attr=edge_attr)
        networkX_graph = to_networkx(graph, 
                                    node_attrs=["x"], 
                                    edge_attrs=["edge_attr"],
                                    graph_attrs=["y"], 
                                    to_undirected=True, 
                                    remove_self_loops=True,
                                    )
        # graph = from_networkx(networkX_graph)
        PyG_graph_list.append(graph)
        networkX_graph_list.append(networkX_graph)
    return PyG_graph_list, networkX_graph_list
