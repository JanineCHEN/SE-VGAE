###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import torch
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F

__all__ = [
    'save_graph_list', 'load_graph_list', 'graph_load_batch',
    'preprocess_graph_list', 'create_graphs'
]

# for one-shot graph generation
def pad_data(a, attr, max_n_node):
    np.fill_diagonal(a, 1.0)  ## fill the diagonal with fill_diag
    max_a = np.zeros([max_n_node, max_n_node])
    max_a[:a.shape[0], :a.shape[1]] = a
    max_a = np.expand_dims(max_a, axis=2)
    zeroes = np.zeros((max_n_node - attr.shape[0]))
    attr = np.concatenate((attr, zeroes))
    attr = np.expand_dims(attr, axis=1)
    return max_a, attr

# for one-shot graph generation
def unpad_data(max_a, attr):
    keep = list()
    max_a = np.reshape(max_a, (max_a.shape[0], max_a.shape[1]))
    max_a[max_a > 0.5] = 1.0
    max_a[max_a <= 0.5] = 0.0
    for i in range(0, max_a.shape[0]):
        if max_a[i][i] > 0:
            keep.append(i)
    ## delete rows and columns
    a = max_a
    a = a[:, keep]  # keep columns
    a = a[keep, :]  # keep rows
    attr = np.reshape(attr, (attr.shape[0]))
    attr = attr[:len(keep)]  ## shorten
    return a, attr


# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)


def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  # adj_list = G.adjacency_list()
  # for id,adj in enumerate(adj_list):
  #     id_min = min(adj)
  #     if id<id_min and id>=1:
  #     # if id<id_min and id>=4:
  #         break
  # node_list = list(range(id)) # only include node prior than node "id"

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list)
  G = max(nx.connected_component_subgraphs(G), key=len)
  return G


def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def preprocess_graph_list(graph_list):
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def graph_load_batch(data_dir,
                     min_num_nodes=2,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     edge_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  if edge_attributes:
    data_edge_att = np.loadtxt(
        os.path.join(path, '{}_edge_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  return graphs


def create_graphs(graph_type, data_dir='data'):
  ### load datasets
  graphs = []
  if graph_type == 'ENZYMES':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=0,
        max_num_nodes=10000,
        name='ENZYMES',
        node_attributes=True,
        graph_labels=True)

  num_nodes = [gg.number_of_nodes() for gg in graphs]
  num_edges = [gg.number_of_edges() for gg in graphs]
  print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
  print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
   
  return graphs

