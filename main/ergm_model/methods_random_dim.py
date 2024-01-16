import copy
from typing import List, Optional

import numpy as np

from ergm_model.ergm_utils import get_neuron_number_from_feature_name, \
    get_neurons_for_distance_between_input_neurons_from_feature_name, get_between_layers_idx_from_feature_name
from ergm_model.methods_base import MethodsBase


class MethodsRandomDim(MethodsBase):
    def __init__(
            self,
            num_neurons: Optional[int],
            output_size: Optional[int],
            possible_dims: List[List[int]],
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.output_neurons = list(range(self.num_neurons))[output_size]
        self.all_possible_max_allowed_connections = {
            tuple(dim): self._max_possible_connections_from_dim(
                dimensions=dim,
            )
            for dim in possible_dims
        }

    def out_degree(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name(
                feature_name=feature_name,
                dimensions=dimensions,
            )
        return sum(graph[neuron])

    def in_degree(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name(
                feature_name=feature_name,
                dimensions=dimensions,
            )
        return sum(graph[:, neuron])

    def entropy(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> float:
        in_degrees = [
            self.in_degree(
                graph=graph,
                feature_name=None,
                neuron=neuron,
                dimensions=dimensions,
            )
            for neuron in range(self.num_neurons)
        ]
        prob_list = [
            neuron_in_degree / sum(in_degrees)
            for neuron_in_degree in in_degrees
            if neuron_in_degree != 0
        ]
        return -(prob_list * np.log(np.abs(prob_list))).sum()

    @staticmethod
    def _max_possible_connections_from_dim(
            dimensions: Optional[List[int]] = None,
    ):
        max_possible_connections = sum(
            dimensions[i] * dimensions[i + 1]
            for i in range(len(dimensions) - 1)
        )
        return max_possible_connections

    def max_possible_connections(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ):
        return self.all_possible_max_allowed_connections[dimensions]

    def distance_between_neurons(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ):
        neuron_a, neuron_b = get_neurons_for_distance_between_input_neurons_from_feature_name(
            dimensions=dimensions,
            feature_name=feature_name,
        )
        if self.out_degree(
                graph=graph,
                feature_name=None,
                neuron=neuron_a,
                dimensions=dimensions,
        ) == 0:
            return -1
        if self.out_degree(
                graph=graph,
                feature_name=None,
                neuron=neuron_b,
                dimensions=dimensions,
        ) == 0:
            return -1
        original_undirected_graph = graph + graph.transpose()
        undirected_graph = copy.deepcopy(original_undirected_graph)
        for i in range((len(dimensions) - 1) * 2):
            undirected_graph = np.matmul(original_undirected_graph, undirected_graph)
            if undirected_graph[neuron_a, neuron_b] != 0:
                return ((i + 2) + 1) / 2
        return -1

    @staticmethod
    def num_paths_from_input_to_output(
            graph: np.ndarray,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name(
                feature_name=feature_name,
                dimensions=dimensions,
            )
        num_paths_in_num_layers_length = np.linalg.matrix_power(graph, len(dimensions) - 1)
        return sum(num_paths_in_num_layers_length[neuron])

    def num_involved_neuron_in_paths_from_input_to_output(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name(
                feature_name=feature_name,
                dimensions=dimensions,
            )
        iterated_neurons = set()
        involved_neurons = {neuron}
        while iterated_neurons != involved_neurons:
            for neuron in involved_neurons - iterated_neurons:
                involved_neurons = involved_neurons.union(set(np.where(graph[neuron] == 1)[0]))
                iterated_neurons = iterated_neurons.union({neuron})
        unconnected_neurons = []
        for potential_n in involved_neurons:
            if potential_n in self.output_neurons:
                continue
            if self.out_degree(
                    graph=graph,
                    feature_name=None,
                    neuron=potential_n,
                    dimensions=dimensions,
            ) == 0:
                unconnected_neurons.append(potential_n)
        return len(involved_neurons - set(unconnected_neurons))

    @staticmethod
    def total_connectivity_between_layers(
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> float:
        between_layers_idx = get_between_layers_idx_from_feature_name(
            feature_name=feature_name,
        )
        first_layer_dim, second_layer_dim = dimensions[between_layers_idx:between_layers_idx + 2]
        max_possible_connections_between_layers = first_layer_dim * second_layer_dim
        block_start_ind = sum(dimensions[:between_layers_idx])
        block_end_ind = block_start_ind + first_layer_dim + second_layer_dim
        connections_between_dims = sum(
            graph[block_start_ind: block_end_ind, block_start_ind: block_end_ind].reshape(-1))
        return round(connections_between_dims / max_possible_connections_between_layers, 2)

    @staticmethod
    def max_connectivity_between_layers(
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> float:
        between_layers_idx = get_between_layers_idx_from_feature_name(
            feature_name=feature_name,
        )
        first_layer_dim, second_layer_dim = dimensions[between_layers_idx:between_layers_idx + 2]
        block_start_ind = sum(dimensions[:between_layers_idx])
        block_end_ind = block_start_ind + first_layer_dim + second_layer_dim
        connections_between_dims = np.sum(graph[block_start_ind: block_end_ind, block_start_ind: block_end_ind], axis=1)
        return round(max(connections_between_dims) / second_layer_dim, 2)

    def modularity(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ):
        raise NotImplementedError

    def layer_connectivity_rank(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ):
        raise NotImplementedError

    def normed_entropy(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ):
        num_connections = self.n_edges(graph, feature_name)
        if num_connections <= 1:
            return 0
        entropy = self.entropy(graph, feature_name)
        return entropy / np.log(
            min(num_connections, sum(dimensions))
        )

    def dimensions(
            self,
            graph,
            feature_name: Optional[str],
            dimensions: Optional[List[int]] = None,
            *args,
    ) -> int:
        layer_idx = get_between_layers_idx_from_feature_name(
            feature_name=feature_name,
        )
        return dimensions[layer_idx]
