from typing import Union, List, Dict, Tuple

import networkx as nx
import numpy as np

from organism import Organism

Logical = Union[bool, int]
COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']


def network_to_adjacency_matrix(
        network: List[np.ndarray],
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    num_neurons = 0
    for layer in network:
        num_neurons += layer.shape[0]
    adjacency_matrix = np.zeros((num_neurons, num_neurons))
    start_in_row = 0
    start_in_col = 0
    neurons_layers_mapping = {}
    for i in range(len(network) - 1):
        layer = network[i]
        num_neurons_in_layer, num_neurons_in_next_layer = layer.shape
        neurons_layers_mapping[i] = [start_in_row + num for num in range(num_neurons_in_layer)]
        end_in_row = start_in_row + num_neurons_in_layer
        start_in_col += num_neurons_in_layer
        end_in_col = start_in_col + num_neurons_in_next_layer
        adjacency_matrix[start_in_row:end_in_row, start_in_col:end_in_col] += layer
        start_in_row = end_in_row
    neurons_layers_mapping[i + 1] = [start_in_row + num for num in range(num_neurons_in_next_layer)]
    return adjacency_matrix, neurons_layers_mapping


def remove_unconnected_neurones(
        adjacency_matrix: np.ndarray,
        input_size: int,
) -> np.ndarray:
    # remove_output_if_doesnt_get_any_thing
    if np.size(np.where(~adjacency_matrix[:, -2:].any(axis=0))[0]) != 0:
        ind_col_to_delete = adjacency_matrix.shape[1] + np.where(~adjacency_matrix[:, -2:].any(axis=0))[0] - 2
        adjacency_matrix = np.delete(adjacency_matrix, ind_col_to_delete, 0)
        adjacency_matrix = np.delete(adjacency_matrix, ind_col_to_delete, 1)
        end_at = -len(ind_col_to_delete)
    else:
        end_at = -2
    while np.size(np.where(~adjacency_matrix[:end_at].any(axis=1))[0]) != 0:
        ind_row_to_delete = np.where(~adjacency_matrix[:end_at].any(axis=1))[0]
        adjacency_matrix = np.delete(adjacency_matrix, ind_row_to_delete, 0)
        adjacency_matrix = np.delete(adjacency_matrix, ind_row_to_delete, 1)
        while np.size(np.where(~adjacency_matrix[:, input_size:].any(axis=0))[0]) != 0:
            ind_col_to_delete = np.where(~adjacency_matrix[:, input_size:].any(axis=0))[0] + input_size
            adjacency_matrix = np.delete(adjacency_matrix, ind_col_to_delete, 0)
            adjacency_matrix = np.delete(adjacency_matrix, ind_col_to_delete, 1)
    return adjacency_matrix


def separate_to_sides(
        organism: Organism,
) -> Tuple[Dict[int, List[int]], List[int], List[int]]:
    neuron_count = 0
    mapping_dict = {}
    new_left_community = []
    new_right_community = []
    for layer_num, layer in enumerate(organism.layers):
        first_community_row_ind = []
        second_community_row_ind = []
        for neuron in range(layer.shape[0]):
            if neuron_count in organism.communities[0]:
                first_community_row_ind.append(neuron)
            else:
                second_community_row_ind.append(neuron)
            neuron_count += 1
        new_left_community += [
            i + max(new_left_community + new_right_community) + 1 if (new_right_community or new_left_community) else i
            for i in range(len(first_community_row_ind))
        ]
        new_right_community += [
            i + max(new_left_community + new_right_community) + 1 if (new_right_community or new_left_community) else i
            for i in range(len(second_community_row_ind))
        ]
        mapping_dict[layer_num] = first_community_row_ind + second_community_row_ind
    return mapping_dict, new_left_community, new_right_community


def rearrange_network_by_module(
        organism: Organism,
        mapping_dict,
):
    for layer_ind in sorted(mapping_dict.keys(), reverse=True):
        if mapping_dict[layer_ind]:
            idx = np.array(mapping_dict[layer_ind])
            if layer_ind > 0:
                organism.layers[layer_ind - 1][:] = organism.layers[layer_ind - 1][:, idx]
            organism.layers[layer_ind][:] = organism.layers[layer_ind][idx, :]


def remove_unused_nodes_from_array(
        organism_layers: List[np.ndarray],
) -> List[np.ndarray]:
    row_ind_to_delete = {
        i: []
        for i in range(len(organism_layers))
    }
    for i in range(len(organism_layers) - 2, -1, -1):
        current_layer = organism_layers[i]
        nodes_with_no_output_current_layer = np.where(~current_layer.any(axis=1))[0].tolist()
        nodes_with_no_input_in_next_layer = np.where(~current_layer.any(axis=0))[0].tolist()
        row_ind_to_delete[i] += nodes_with_no_output_current_layer
        row_ind_to_delete[i + 1] += nodes_with_no_input_in_next_layer
    new_layers = []
    for layer, nodes in row_ind_to_delete.items():
        new_layers.append(np.delete(organism_layers[layer], nodes, axis=0))
        if layer > 0:
            new_layers[layer - 1] = np.delete(new_layers[layer - 1], nodes, axis=1)
    return new_layers


def get_full_network_from_list_of_layers(
        list_of_layers: List[nx.DiGraph],
) -> nx.DiGraph:
    n = list_of_layers[0]
    for i in range(len(list_of_layers) - 1):
        n = nx.compose(n, list_of_layers[i + 1])
    return n
