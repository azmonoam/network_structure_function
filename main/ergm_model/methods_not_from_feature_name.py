import copy
from typing import List

import numpy as np
import numpy.ma as ma

from ergm_utils import creat_ff_fully_connected_mask


class methods:
    def __init__(
            self,
            dimensions: List[int],
    ):
        self.num_neurons = sum(dimensions)
        self.dimensions = dimensions
        self.fully_connected_mask_array = creat_ff_fully_connected_mask(
            num_neurons=self.num_neurons,
            dimensions=self.dimensions,
        )
        self.max_possible_connections = self.fully_connected_mask_array.sum()

    @staticmethod
    def out_degree(
            graph: np.ndarray,
            neuron: int
    ) -> int:
        return sum(graph[neuron])

    @staticmethod
    def in_degree(
            graph: np.ndarray,
            neuron: int
    ) -> int:
        return sum(graph[:, neuron])

    @staticmethod
    def n_edges(
            graph: np.ndarray,
    ) -> int:
        return sum(graph.reshape(-1))

    def entropy(
            self,
            graph: np.ndarray,
    ) -> float:
        in_degrees = [
            self.in_degree(
                graph=graph,
                neuron=neuron,
            )
            for neuron in range(self.num_neurons)
        ]
        prob_list = [
            neuron_in_degree / sum(in_degrees)
            for neuron_in_degree in in_degrees
            if neuron_in_degree != 0
        ]
        return -(prob_list * np.log(np.abs(prob_list))).sum()

    def sum_edges_out_of_ff_structure(
            self,
            graph: np.ndarray,
    ):
        masked_graph = ma.masked_array(graph, mask=self.fully_connected_mask_array)
        return masked_graph.sum()

    def max_possible_connections(
            self,
    ):
        return self.max_possible_connections

    def density(
            self,
            graph: np.ndarray,
    ):
        return round(self.n_edges(graph) / self.max_possible_connections, 2)

    def distance_between_neurons(
            self,
            neurons: List[int],
            graph: np.ndarray,
    ):
        original_undirected_graph = graph + graph.transpose()
        undirected_graph = copy.deepcopy(original_undirected_graph)
        for i in range((len(self.dimensions) - 1) * 2):
            undirected_graph = np.matmul(original_undirected_graph, undirected_graph)
            if undirected_graph[neurons[0], neurons[1]] != 0:
                return ((i + 2) + 1) / 2
        return -1

    def num_paths_from_input_to_output(
            self,
            neuron: int,
            graph: np.ndarray,
    ) -> int:
        num_paths_in_num_layers_length = np.linalg.matrix_power(graph, len(self.dimensions) - 1)
        return sum(num_paths_in_num_layers_length[neuron])

    def num_involved_neuro_in_paths_from_input_to_output(
            self,
            neuron: int,
            graph: np.ndarray,
    ) -> int:
        iterated_neurons = set()
        involved_neurons = {neuron}
        while iterated_neurons != involved_neurons:
            for neuron in involved_neurons - iterated_neurons:
                involved_neurons = involved_neurons.union(set(np.where(graph[neuron] == 1)[0]))
                iterated_neurons = iterated_neurons.union({neuron})
        return len(involved_neurons)

    def total_connectivity_between_layers(
            self,
            between_layers_idx: int,
            graph: np.ndarray,
    ) -> float:
        first_layer_dim, second_layer_dim = self.dimensions[between_layers_idx:between_layers_idx + 2]
        max_possible_connections_between_layers = first_layer_dim * second_layer_dim
        block_start_ind = sum(self.dimensions[:between_layers_idx])
        block_end_ind = block_start_ind + first_layer_dim + second_layer_dim
        connections_between_dims = sum(
            graph[block_start_ind: block_end_ind, block_start_ind: block_end_ind].reshape(-1))
        return round(connections_between_dims / max_possible_connections_between_layers, 2)

    def max_connectivity_between_layers(
            self,
            between_layers_idx: int,
            graph: np.ndarray,
    ) -> float:
        first_layer_dim, second_layer_dim = self.dimensions[between_layers_idx:between_layers_idx + 2]
        block_start_ind = sum(self.dimensions[:between_layers_idx])
        block_end_ind = block_start_ind + first_layer_dim + second_layer_dim
        connections_between_dims = np.sum(graph[block_start_ind: block_end_ind, block_start_ind: block_end_ind], axis=1)
        return round(max(connections_between_dims) / second_layer_dim, 2)
