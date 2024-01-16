from typing import Optional

import numpy as np
from scipy.special import comb


class MethodsBase:
    def __init__(
            self,
    ):
        self.method_features_mapping = {
            'connectivity_ratio': self.density,
            'distances_between_input_neuron': self.distance_between_neurons,
            'entropy': self.entropy,
            'in_connections_per_layer': self.in_degree,
            'layer_connectivity_rank': self.layer_connectivity_rank,
            'max_connectivity_between_layers_per_layer': self.max_connectivity_between_layers,
            'max_possible_connections': self.max_possible_connections,
            'modularity': self.modularity,
            'normed_entropy': self.normed_entropy,
            'num_connections': self.n_edges,
            'num_involved_neurons_in_paths_per_input_neuron': self.num_involved_neuron_in_paths_from_input_to_output,
            'num_paths_to_output_per_input_neuron': self.num_paths_from_input_to_output,
            'out_connections_per_layer': self.out_degree,
            'total_connectivity_ratio_between_layers': self.total_connectivity_between_layers,
            'idegree_2': self.idegree_2,
            'motifs_count_0': self.motifs_count_0,
            'motifs_count_1': self.motifs_count_1,
            'motifs_count_2': self.motifs_count_2,
            'num_isolated_neurones': self.num_isolated_neurones,
            'dimensions': self.dimensions,
        }

    def out_degree(
            self,
            graph,
            feature_name,
            neuron,
            *args,
    ):
        pass

    def in_degree(
            self,
            graph,
            feature_name,
            neuron,
            *args,
    ):
        pass

    @staticmethod
    def n_edges(
            graph: np.ndarray,
            feature_name: Optional[str],
            *args,
    ) -> int:
        return np.sum(graph.reshape(-1)).item()

    def entropy(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def max_possible_connections(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def density(
            self,
            graph,
            feature_name,
            *args,
    ):
        max_possible_connections = self.max_possible_connections(
            graph=graph,
            feature_name=None,
            *args,
        )
        return round(self.n_edges(graph, feature_name) / max_possible_connections, 2)

    def distance_between_neurons(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def num_paths_from_input_to_output(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def num_involved_neuron_in_paths_from_input_to_output(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def total_connectivity_between_layers(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def max_connectivity_between_layers(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def modularity(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def layer_connectivity_rank(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def normed_entropy(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass

    def idegree_2(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            *args,
    ):
        return sum([1 for i in np.sum(graph, axis=0) if i == 2])

    def motifs_count_0(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            *args,
    ):
        return sum(comb(np.sum(graph, axis=1), 2))

    def motifs_count_1(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            *args,
    ):
        return np.matmul(np.sum(graph, axis=0), np.sum(graph, axis=1))

    def motifs_count_2(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            *args,
    ):
        return sum(comb(np.sum(graph, axis=0), 2))

    def num_isolated_neurones(
            self,
            graph: np.ndarray,
            feature_name: Optional[str],
            *args,
    ):
        return sum(np.sum(graph, axis=0) + np.sum(graph, axis=1) == 0)

    def dimensions(
            self,
            graph,
            feature_name,
            *args,
    ):
        pass
