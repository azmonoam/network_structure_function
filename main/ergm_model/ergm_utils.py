import re
from copy import deepcopy
from itertools import combinations
from typing import List, Tuple

import numpy as np
from utils.set_up_organism_utils import NodeType


def switch_edge(
        graph: np.ndarray,
        i: int,
        j: int,
):
    new_g = deepcopy(graph)
    if new_g[i, j] == 1:
        new_g[i, j] = 0
    else:
        new_g[i, j] = 1
    return new_g


def creat_ff_fully_connected_mask(
        num_neurons: int,
        dimensions: List[int],
) -> np.ndarray:
    fully_connected_ff_mask = np.zeros((num_neurons, num_neurons))
    for layer in range(len(dimensions) - 1):
        start_ind = sum(dimensions[:layer])
        ff_block_min_row = start_ind
        ff_block_max_row = start_ind + dimensions[layer]
        ff_block_min_column = start_ind + dimensions[layer]
        ff_block_max_column = start_ind + dimensions[layer] + dimensions[layer + 1]
        fully_connected_ff_mask[ff_block_min_row:ff_block_max_row, ff_block_min_column:ff_block_max_column] = 1
    return fully_connected_ff_mask


def get_allowed_connections_based_on_ff_network_dims(
        dimensions: List[int],
) -> List[Tuple[int, int]]:
    allowed_connections = []
    for d in range(len(dimensions) - 1):
        for i in range(sum(dimensions[:d]), sum(dimensions[:d]) + dimensions[d]):
            for j in range(sum(dimensions[:d + 1]), sum(dimensions[:d + 1]) + dimensions[d + 1]):
                allowed_connections.append((i, j))
    return allowed_connections


def get_neuron_number_from_feature_name(
        feature_name: str,
        dimensions: List[int],
) -> int:
    layer_ind, neuron_number = (int(num) for num in re.findall(r'\d', feature_name))
    return sum(dimensions[:layer_ind]) + neuron_number


def get_neurons_for_distance_between_input_neurons_from_feature_name(
        feature_name: str,
        dimensions: List[int],
) -> Tuple[int, int]:
    neurons_combination_list = list(combinations(range(dimensions[0]), 2))
    combination_ind = int(re.search(r'\d+', feature_name).group())
    return neurons_combination_list[combination_ind]


def get_between_layers_idx_from_feature_name(
        feature_name: str,
) -> int:
    return int(re.search(r'\d+', feature_name).group())


def get_neurons_for_distance_between_input_neurons_from_feature_name_nx(
        feature_name: str,
        layer_neuron_idx_mapping,
) -> Tuple[int, int]:
    neurons_combination_list = list(combinations(layer_neuron_idx_mapping[0], 2))
    combination_ind = int(re.search(r'\d+', feature_name).group())
    return neurons_combination_list[combination_ind]


def get_neuron_number_from_feature_name_nx(
        feature_name: str,
) -> NodeType:
    layer_ind, neuron_number = (int(num) for num in re.findall(r'\d+', feature_name))
    return (layer_ind, neuron_number)
