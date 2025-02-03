import os
import random
import json
import pickle
import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data
import networkx as nx
from networkx.readwrite import json_graph
import multiprocessing as mp
from torch.optim.lr_scheduler import CyclicLR

from .config import get_config
cfg = get_config()
if cfg.device == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cfg.device == 'cpu':
    device = 'cpu'

def load_checkpoint(ckpt_dir_or_file, map_location=torch.device('cpu')):
    ckpt = torch.load(ckpt_dir_or_file, map_location=map_location)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
    return ckpt

def load_model(model_path, model):
    ckpt_path = model_path + f'/FP_VAE_{cfg.resume_ep}.pth'
    try:
        ckpt = load_checkpoint(ckpt_path)
        start_ep = ckpt['epoch'] + 1
        print(f'Try to load checkpoint of epoch {start_ep-1}!')
        ckpt['model'] = {key.replace("module.", ""): value for key, value in ckpt['model'].items()}
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        if cfg.lr_schedule:
            scheduler = CyclicLR(optimizer,
                                    base_lr=cfg.base_lr,
                                    max_lr=cfg.max_lr,
                                    step_size_up=cfg.step_size_up,
                                    mode="triangular",
                                    cycle_momentum=False)
            print("CyclicLR is adopted!!!")
        else:
            scheduler = None
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f' Checkpoint of epoch {start_ep-1} loaded!')
    except:
        print(' [*] No checkpoint!')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        if cfg.lr_schedule:
            scheduler = CyclicLR(optimizer,
                                    base_lr=cfg.base_lr,
                                    max_lr=cfg.max_lr,
                                    step_size_up=cfg.step_size_up,
                                    mode="triangular",
                                    cycle_momentum=False)
            print("CyclicLR is adopted!!!")
        else:
            scheduler = None
        start_ep = 0
    return start_ep, model, optimizer, scheduler


def Graph_load_batch(min_num_nodes = 2, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True, graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'data/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    # max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        # print(nodes)
        G_sub_ = G.subgraph(nodes)
        G_sub = G_sub_.copy()
        # print(G_sub)
        if graph_labels:
            # G_sub.graph['label'] = data_graph_labels[i]
            G_sub.graph['y'] = data_graph_labels[i]
        # print(G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            # print(len(graphs))
            # print(graphs[0])
            # print(graphs[0].nodes())
            # print(graphs[0].graph)
            # if G_sub.number_of_nodes() > max_nodes:
            #     max_nodes = G_sub.number_of_nodes()
    print('Loaded')
    return graphs, data_node_att, data_node_label

# main data load function
def load_graphs(min_num_nodes = 2, max_num_nodes = 1000, name = 'ENZYMES'):
    node_labels = [None]
    edge_labels = [None]
    idx_train = [None]
    idx_val = [None]
    idx_test = [None]

    try:
        graphs_all, node_features_all, node_labels_all = Graph_load_batch(min_num_nodes = min_num_nodes, 
                                                                max_num_nodes = max_num_nodes, 
                                                                name = name)
        node_features_all = (node_features_all-np.mean(node_features_all,axis=-1,keepdims=True))/np.std(node_features_all,axis=-1,keepdims=True)
        graphs = []
        node_labels = []
        node_features = []
        edge_labels = []
        for graph in graphs_all:
            n = graph.number_of_nodes()
            label = np.zeros((n, n),dtype=int)
            # what are we doing here?
            for i,u in enumerate(graph.nodes()):
                for j,v in enumerate(graph.nodes()):
                    if node_labels_all[u-1] == node_labels_all[v-1] and u>v:
                        label[i,j] = 1
            # if label.sum() > n*n/4:
            #     continue
            edge_labels.append(label)

            graphs.append(graph)

            idx = [node-1 for node in graph.nodes()]
            node_label = node_labels_all[idx]
            node_labels.append(node_label)
            node_feature = node_features_all[idx,:]
            node_features.append(node_feature)

        print('final num', len(graphs))

    except:
        raise NotImplementedError

    return graphs, node_features, edge_labels, node_labels, idx_train, idx_val, idx_test


def nx_to_pyg_data(graphs, node_features, edge_labels=None):
    data_list = []
    for i in range(len(graphs)):
        node_feature = node_features[i]
        graph = graphs[i].copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # relabel graphs
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(graph, mapping, copy=False)

        x = np.zeros(node_feature.shape)
        graph_nodes = list(graph.nodes)
        for m in range(node_feature.shape[0]):
            x[graph_nodes[m]] = node_feature[m]
        x = torch.from_numpy(x).float()

        # get edges
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1,0)

        # get graph labels
        # y = torch.tensor(graph.graph['label'], dtype=torch.int8)
        y = torch.tensor(graph.graph['y'], dtype=torch.int8)

        data = Data(x=x, edge_index=edge_index, y=y)

        # get edge_labels
        if edge_labels[0] is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive

        data_list.append(data)
    return data_list


def load_pyg_dataset(min_num_nodes = 2, 
                    max_num_nodes = 1000, 
                    name = 'ENZYMES'):
    graphs, node_features, edge_labels, node_labels,_,_,_ = load_graphs(min_num_nodes = min_num_nodes, 
                                                        max_num_nodes = max_num_nodes, 
                                                        name = name)
    return nx_to_pyg_data(graphs, node_features, edge_labels)


def deduplicate_edges(edges):
    edges_new = np.zeros((2,edges.shape[1]//2), dtype=int)
    # add none self edge
    j = 0
    skip_node = {} # node already put into result
    for i in range(edges.shape[1]):
        if edges[0,i]<edges[1,i]:
            edges_new[:,j] = edges[:,i]
            j += 1
        elif edges[0,i]==edges[1,i] and edges[0,i] not in skip_node:
            edges_new[:,j] = edges[:,i]
            skip_node.add(edges[0,i])
            j += 1

    return edges_new


def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negtive_edges):
    mask_link_positive_set = []
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:,i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = np.zeros((2,num_negtive_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negtive_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:,i] = mask_temp
                break

    return mask_link_negative


def resample_edge_mask_link_negative(data):
    data.mask_link_negative_train = get_edge_mask_link_negative(data.mask_link_positive_train, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_train.shape[1])
    data.mask_link_negative_val = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_val.shape[1])
    data.mask_link_negative_test = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                     num_negtive_edges=data.mask_link_positive_test.shape[1])


def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        index_test = index_val[:len(index_val)//2]
        index_val = index_val[len(index_val)//2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        split2 = int((1-remove_ratio/2)*e)
        edges_train = edges[:,:split1]
        edges_val = edges[:,split1:split2]
        edges_test = edges[:,split2:]

    return edges_train, edges_val, edges_test


def get_link_mask(data, remove_ratio=0.2, resplit=True, infer_link_positive=True):
    if resplit:
        if infer_link_positive:
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train, data.mask_link_positive_val, data.mask_link_positive_test = \
            split_edges(data.mask_link_positive, remove_ratio)
    resample_edge_mask_link_negative(data)


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array


def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1,:]), axis=-1)


def get_pyg_dataset(dataset_name, 
                    min_num_nodes = 2, 
                    max_num_nodes = 1000,
                    remove_link_ratio=0.2, 
                    approximate=-1, 
                    # use_cache=True, 
                    use_cache=False, 
                    # remove_feature=False, 
                    # task='link',
                    task='',
                    ):
    try:
        dataset = load_pyg_dataset(min_num_nodes = min_num_nodes, 
                                    max_num_nodes = max_num_nodes, 
                                    name = dataset_name)
    except:
        raise NotImplementedError

    # precompute shortest path
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    if not os.path.isdir('datasets/cache'):
        os.mkdir('datasets/cache')
    f1_name = 'datasets/cache/' + dataset_name + str(approximate) + '_dists.dat'
    f2_name = 'datasets/cache/' + dataset_name + str(approximate)+ '_dists_removed.dat'
    f3_name = 'datasets/cache/' + dataset_name + str(approximate)+ '_links_train.dat'
    f4_name = 'datasets/cache/' + dataset_name + str(approximate)+ '_links_val.dat'
    f5_name = 'datasets/cache/' + dataset_name + str(approximate)+ '_links_test.dat'

    if use_cache and ((os.path.isfile(f2_name) and task=='link') or (os.path.isfile(f1_name) and task!='link')):
        with open(f3_name, 'rb') as f3, \
            open(f4_name, 'rb') as f4, \
            open(f5_name, 'rb') as f5:
            links_train_list = pickle.load(f3)
            links_val_list = pickle.load(f4)
            links_test_list = pickle.load(f5)
        if task=='link':
            with open(f2_name, 'rb') as f2:
                dists_removed_list = pickle.load(f2)
        else:
            with open(f1_name, 'rb') as f1:
                dists_list = pickle.load(f1)

        print('Cache loaded!')
        data_list = []
        for i, data in enumerate(dataset):
            if task == 'link':
                data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
            data.mask_link_positive_train = links_train_list[i]
            data.mask_link_positive_val = links_val_list[i]
            data.mask_link_positive_test = links_test_list[i]
            get_link_mask(data, resplit=False)

            if task=='link':
                data.dists = torch.from_numpy(dists_removed_list[i]).float()
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()
            else:
                data.dists = torch.from_numpy(dists_list[i]).float()
            # if remove_feature:
            #     data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)
    else:
        data_list = []
        dists_list = []
        dists_removed_list = []
        links_train_list = []
        links_val_list = []
        links_test_list = []
        for i, data in enumerate(dataset):
            if 'link' in task:
                get_link_mask(data, remove_link_ratio, resplit=True,
                              infer_link_positive=True if task == 'link' else False)
                links_train_list.append(data.mask_link_positive_train)
                links_val_list.append(data.mask_link_positive_val)
                links_test_list.append(data.mask_link_positive_test)
            if task=='link':
                dists_removed = precompute_dist_data(data.mask_link_positive_train, data.num_nodes,
                                                     approximate=approximate)
                dists_removed_list.append(dists_removed)
                data.dists = torch.from_numpy(dists_removed).float()
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()

            else:
                dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=approximate)
                dists_list.append(dists)
                data.dists = torch.from_numpy(dists).float()
            # if remove_feature:
            #     data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)

        with open(f1_name, 'wb') as f1, \
            open(f2_name, 'wb') as f2, \
            open(f3_name, 'wb') as f3, \
            open(f4_name, 'wb') as f4, \
            open(f5_name, 'wb') as f5:

            if task=='link':
                pickle.dump(dists_removed_list, f2)
            else:
                pickle.dump(dists_list, f1)
            pickle.dump(links_train_list, f3)
            pickle.dump(links_val_list, f4)
            pickle.dump(links_test_list, f5)
        print('Cache saved!')

    return data_list


def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id


def get_dist_max(anchorset_id, dist, device='cpu'):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


# preselect anchor set for each graph in the graph dataset
def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):

    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num//anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2**(i+1)-1
        anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes,c=1)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)