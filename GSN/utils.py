import torch
import numpy as np
from torch_geometric.utils import remove_self_loops

import torch
import numpy as np
from torch_geometric.utils import remove_self_loops, to_undirected
import networkx as nx

import graph_tool as gt
import graph_tool.topology as gt_topology

import os
import glob
import re
import torch
from torch_geometric.data import Data
import networkx as nx
import types

import torch_geometric as torch_geo

def GSN_prepare(data, subgraph_dicts,
             subgraph_counts2ids, cnt_fn, orbit_partition_sizes, args):
    new_data = Data()
    setattr(new_data, 'edge_index', torch.from_numpy(np.transpose(data['edges'])))
    setattr(new_data, 'x', torch.from_numpy(data['node_features']))
    setattr(new_data, 'graph_size', torch.from_numpy(data['node_features']).shape[0])
    if new_data.edge_index.shape[1] == 0:
        setattr(new_data, 'degrees', torch.zeros((new_data.graph_size,)))
    else:
        setattr(new_data, 'degrees', torch_geo.utils.degree(new_data.edge_index[0].long()))
    # print("#####: ", new_data)
    # if hasattr(data, 'edge_features'):
    setattr(new_data, 'edge_features', torch.from_numpy(data['edge_features']))    
    # setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).float())
    setattr(new_data, 'y', torch.tensor(data['target'][0]).unsqueeze(0).long())
    if new_data.edge_index.shape[1] == 0 and cnt_fn.__name__ == 'subgraph_isomorphism_edge_counts':
        setattr(new_data, 'identifiers', torch.zeros((0, sum(orbit_partition_sizes))).long())
    else:
        new_data = subgraph_counts2ids(cnt_fn, new_data, subgraph_dicts, args)
    return new_data

def automorphism_orbits(edge_list, print_msgs=True, **kwargs):
    
    ##### vertex automorphism orbits ##### 

    directed=kwargs['directed'] if 'directed' in kwargs else False
    
    graph = gt.Graph(directed=directed)
    graph.add_edge_list(edge_list)
    # gt.generation.remove_self_loops(graph)
    # gt.generation.remove_parallel_edges(graph)
    gt.generation.remove_self_loops(graph)
    gt.generation.remove_parallel_edges(graph)

    # compute the vertex automorphism group
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v
    
    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role
        
    orbit_membership_list = [[],[]]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse = True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i,vertex in enumerate(orbit_membership_list[0])}


    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit]+[vertex]
    
    aut_count = len(aut_group)
    
    if print_msgs:
        print('Orbit partition of given substructure: {}'.format(orbit_partition)) 
        print('Number of orbits: {}'.format(len(orbit_partition)))
        print('Automorphism count: {}'.format(aut_count))

    return graph, orbit_partition, orbit_membership, aut_count

def induced_edge_automorphism_orbits(edge_list, **kwargs):
    
    ##### induced edge automorphism orbits (according to the vertex automorphism group) #####
    
    directed=kwargs['directed'] if 'directed' in kwargs else False
    directed_orbits=kwargs['directed_orbits'] if 'directed_orbits' in kwargs else False

    graph, orbit_partition, orbit_membership, aut_count = automorphism_orbits(edge_list=edge_list,
                                                                              directed=directed,
                                                                              print_msgs=False)
    edge_orbit_partition = dict()
    edge_orbit_membership = dict()
    edge_orbits2inds = dict()
    ind = 0
    
    if not directed:
        edge_list = to_undirected(torch.tensor(graph.get_edges()).transpose(1,0)).transpose(1,0).tolist()

    # infer edge automorphisms from the vertex automorphisms
    for i,edge in enumerate(edge_list):
        if directed_orbits:
            edge_orbit = (orbit_membership[edge[0]], orbit_membership[edge[1]])
        else:
            edge_orbit = frozenset([orbit_membership[edge[0]], orbit_membership[edge[1]]])
        if edge_orbit not in edge_orbits2inds:
            edge_orbits2inds[edge_orbit] = ind
            ind_edge_orbit = ind
            ind += 1
        else:
            ind_edge_orbit = edge_orbits2inds[edge_orbit]

        if ind_edge_orbit not in edge_orbit_partition:
            edge_orbit_partition[ind_edge_orbit] = [tuple(edge)]
        else:
            edge_orbit_partition[ind_edge_orbit] += [tuple(edge)] 

        edge_orbit_membership[i] = ind_edge_orbit

    print('Edge orbit partition of given substructure: {}'.format(edge_orbit_partition)) 
    print('Number of edge orbits: {}'.format(len(edge_orbit_partition)))
    print('Graph (vertex) automorphism count: {}'.format(aut_count))
    
    return graph, edge_orbit_partition, edge_orbit_membership, aut_count


def subgraph_isomorphism_vertex_counts(edge_index, **kwargs):
    
    ##### vertex structural identifiers #####
    
    subgraph_dict, induced, num_nodes = kwargs['subgraph_dict'], kwargs['induced'], kwargs['num_nodes']
    directed = kwargs['directed'] if 'directed' in kwargs else False
    
    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index.transpose(1,0).cpu().numpy()))
    gt.generation.remove_self_loops(G_gt)
    gt.generation.remove_parallel_edges(G_gt)  
       
    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True, generator=True)
    
    ## num_nodes should be explicitly set for the following edge case: 
    ## when there is an isolated vertex whose index is larger
    ## than the maximum available index in the edge_index
    
    counts = np.zeros((num_nodes, len(subgraph_dict['orbit_partition'])))
    for sub_iso_curr in sub_iso:
        for i,node in enumerate(sub_iso_curr):
            # increase the count for each orbit
            counts[node, subgraph_dict['orbit_membership'][i]] +=1
    counts = counts/subgraph_dict['aut_count']
        
    counts = torch.tensor(counts)
    
    return counts


def subgraph_isomorphism_edge_counts(edge_index, **kwargs):
    
    ##### edge structural identifiers #####
    
    subgraph_dict, induced = kwargs['subgraph_dict'], kwargs['induced']
    directed = kwargs['directed'] if 'directed' in kwargs else False
    
    edge_index = edge_index.transpose(1,0).cpu().numpy()
    edge_dict = {}
    for i, edge in enumerate(edge_index):         
        edge_dict[tuple(edge)] = i
        
    if not directed:
        subgraph_edges = to_undirected(torch.tensor(subgraph_dict['subgraph'].get_edges().tolist()).transpose(1,0)).transpose(1,0).tolist()

    
    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index))
    gt.generation.remove_self_loops(G_gt)
    gt.generation.remove_parallel_edges(G_gt)  
       
    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True, generator=True)
    
            
    counts = np.zeros((edge_index.shape[0], len(subgraph_dict['orbit_partition'])))
    
    for sub_iso_curr in sub_iso:
        mapping = sub_iso_curr.get_array()
#         import pdb;pdb.set_trace()
        for i,edge in enumerate(subgraph_edges): 
            
            # for every edge in the graph H, find the edge in the subgraph G_S to which it is mapped
            # (by finding where its endpoints are matched). 
            # Then, increase the count of the matched edge w.r.t. the corresponding orbit
            # Repeat for the reverse edge (the one with the opposite direction)
            
            edge_orbit = subgraph_dict['orbit_membership'][i]
            mapped_edge = tuple([mapping[edge[0]], mapping[edge[1]]])
            counts[edge_dict[mapped_edge], edge_orbit] += 1
            
    counts = counts/subgraph_dict['aut_count']
    
    counts = torch.tensor(counts)
    
    return counts


def subgraph_counts2ids(count_fn, data, subgraph_dicts, subgraph_params):
    
    #### Remove self loops and then assign the structural identifiers by computing subgraph isomorphisms ####
    
    if hasattr(data, 'edge_features'):
        edge_index, edge_features = remove_self_loops(data.edge_index, data.edge_features)
        setattr(data, 'edge_features', edge_features)
    else:
        edge_index = remove_self_loops(data.edge_index)[0]
             
    num_nodes = data.x.shape[0]
    identifiers = None
    for subgraph_dict in subgraph_dicts:
        kwargs = {'subgraph_dict': subgraph_dict, 
                  'induced': subgraph_params['induced'],
                  'num_nodes': num_nodes,
                  'directed': subgraph_params['directed']}
        counts = count_fn(edge_index, **kwargs)
        identifiers = counts if identifiers is None else torch.cat((identifiers, counts),1) 
    setattr(data, 'edge_index', edge_index)
    setattr(data, 'identifiers', identifiers.long())
    
    return data


def load_dataset(data_file):
    '''
        Loads dataset from `data_file`.
    '''
    print("Loading dataset from {}".format(data_file))
    dataset_obj = torch.load(data_file)
    graphs_ptg = dataset_obj[0]
    num_classes = dataset_obj[1]
    orbit_partition_sizes = dataset_obj[2]
    return graphs_ptg, num_classes, orbit_partition_sizes

def find_id_filename(data_folder, id_type, induced, directed_orbits, k):
    '''
        Looks for existing precomputed datasets in `data_folder` with counts for substructure 
        `id_type` larger `k`.
    '''
    if induced:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_induced_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_induced_[0-9]*.pt'.format(id_type))
    else:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_[0-9]*.pt'.format(id_type))
    filenames = glob.glob(pattern)
    for name in filenames:
        k_found = int(re.findall(r'\d+', name)[-1])
        if k_found >= k:
            return name, k_found
    return None, None

def downgrade_k(dataset, k, orbit_partition_sizes, k_min):
    '''
        Donwgrades `dataset` by keeping only the orbits of the requested substructures.
    '''
    feature_vector_size = sum(orbit_partition_sizes[0:k-k_min+1])
    graphs_ptg = list()
    for data in dataset:
        new_data = Data()
        for attr in data.__iter__():
            name, value = attr
            setattr(new_data, name, value)
        setattr(new_data, 'identifiers', data.identifiers[:, 0:feature_vector_size])
        graphs_ptg.append(new_data)
    return graphs_ptg, orbit_partition_sizes[0:k-k_min+1]

def try_downgrading(data_folder, id_type, induced, directed_orbits, k, k_min):
    '''
        Extracts the substructures of size up to the `k`, if a collection of substructures
        with size larger than k has already been computed.
    '''
    found_data_filename, k_found = find_id_filename(data_folder, id_type, induced, directed_orbits, k)
    if found_data_filename is not None:
        graphs_ptg, num_classes, orbit_partition_sizes = load_dataset(found_data_filename)
        print("Downgrading k from dataset {}...".format(found_data_filename))
        graphs_ptg, orbit_partition_sizes = downgrade_k(graphs_ptg, k, orbit_partition_sizes, k_min)
        return True, graphs_ptg, num_classes, orbit_partition_sizes
    else:
        return False, None, None, None
    
def get_custom_edge_list(ks, substructure_type=None, filename=None):
    '''
        Instantiates a list of `edge_list`s representing substructures
        of type `substructure_type` with sizes specified by `ks`.
    ''' 
    if substructure_type is None and filename is None:
        raise ValueError('You must specify either a type or a filename where to read substructures from.')
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))
    return edge_lists

def process_arguments(args):
    ###### choose the function that computes the automorphism group and the orbits #######
    if args['edge_automorphism'] == 'induced':
        automorphism_fn = induced_edge_automorphism_orbits if  args['id_scope'] == 'local' else automorphism_orbits
    elif args['edge_automorphism'] == 'line_graph':
        automorphism_fn = edge_automorphism_orbits if  args['id_scope'] == 'local' else automorphism_orbits
    else:
        raise NotImplementedError
    ###### choose the function that computes the subgraph isomorphisms #######
    count_fn = subgraph_isomorphism_edge_counts if args['id_scope'] == 'local'else subgraph_isomorphism_vertex_counts

    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user
    if args['id_type'] in ['cycle_graph',
                           'path_graph',
                           'complete_graph',
                           'binomial_tree',
                           'star_graph',
                           'nonisomorphic_trees']:
        args['k'] = args['k'][0]
        k_max = args['k']
        k_min = 2 if args['id_type'] == 'star_graph' else 3
        args['custom_edge_list'] = get_custom_edge_list(list(range(k_min, k_max + 1)), args['id_type'])         

    elif args['id_type'] in ['cycle_graph_chosen_k',
                             'path_graph_chosen_k', 
                             'complete_graph_chosen_k',
                             'binomial_tree_chosen_k',
                             'star_graph_chosen_k',
                             'nonisomorphic_trees_chosen_k']:
        args['custom_edge_list'] = get_custom_edge_list(args['k'], args['id_type'].replace('_chosen_k',''))
        
    elif args['id_type'] in ['all_simple_graphs']:
        args['k'] = args['k'][0]
        k_max = args['k']
        k_min = 3
        filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        args['custom_edge_list'] = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)
        
    elif args['id_type'] in ['all_simple_graphs_chosen_k']:
        filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        args['custom_edge_list'] = get_custom_edge_list(args['k'], filename=filename)
        
    elif args['id_type'] in ['diamond_graph']:
        args['k'] = None
        graph_nx = nx.diamond_graph()
        args['custom_edge_list'] = [list(graph_nx.edges)]

    elif args['id_type'] == 'custom':
        assert args['custom_edge_list'] is not None, "Custom edge list must be provided."
    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(args['id_type']))
        
    return args, count_fn, automorphism_fn