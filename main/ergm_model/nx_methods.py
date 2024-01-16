from typing import Optional
from networkx.algorithms import isomorphism

import networkx as nx
import numpy as np

from ergm_model.methods_base import MethodsBase
from ergm_model.ergm_utils import (
    get_neurons_for_distance_between_input_neurons_from_feature_name_nx,
    get_neuron_number_from_feature_name_nx,
    get_between_layers_idx_from_feature_name,
)
from new_organism import Organism
from stractural_features_models.modularity_calc import ModularityCalc
from utils.main_utils import get_number_of_connections


class NxMethods(MethodsBase):
    def __init__(
            self,
            connectivity_in_perc: bool = False,
    ):
        super().__init__()
        self.connectivity_in_perc = connectivity_in_perc
        self.method_features_mapping['motifs_count_0'] = self.motifs_count_0_
        self.method_features_mapping['motifs_count_1'] = self.motifs_count_1_
        self.method_features_mapping['motifs_count_2'] = self.motifs_count_2_


    @staticmethod
    def n_edges(
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ) -> int:
        return graph.network.number_of_edges()

    def max_possible_connections(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ) -> int:
        return sum(
            graph.dimensions[i] * graph.dimensions[i + 1]
            for i in range(graph.num_layers)
        )

    def density(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ) -> float:
        num_connections = get_number_of_connections(
            organism=graph,
        )
        max_possible_connections = self.max_possible_connections(
            graph=graph,
            feature_name=None,
        )
        return round(num_connections / max_possible_connections, 2)

    def entropy(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ) -> float:
        deg = [d for n, d in graph.network.in_degree() if n > graph.layer_neuron_idx_mapping[0][-1]]
        prob_list = [i / sum(deg) for i in deg if i != 0]
        return -(prob_list * np.log(np.abs(prob_list))).sum()

    def distance_between_neurons(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ):
        neuron_a, neuron_b = get_neurons_for_distance_between_input_neurons_from_feature_name_nx(
            layer_neuron_idx_mapping=graph.layer_neuron_idx_mapping,
            feature_name=feature_name,
        )
        undirected_org = graph.network.to_undirected()
        if neuron_a not in graph.network.nodes(neuron_a) or neuron_b not in graph.network.nodes(neuron_b):
            return -1
        try:
            distance = len(
                nx.shortest_path(
                    undirected_org,
                    source=neuron_a,
                    target=neuron_b,
                )
            ) / 2
        except nx.exception.NetworkXNoPath:
            distance = -1
        return distance

    def in_degree(
            self,
            graph: Organism,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name_nx(
                feature_name=feature_name,
            )
        return graph.network.in_degree(neuron)

    def out_degree(
            self,
            graph: Organism,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name_nx(
                feature_name=feature_name,
            )
        return graph.network.out_degree(neuron)

    def modularity(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ):
        graph.clean_network = graph.network
        stractural_feature_calc = ModularityCalc(
            organism=graph,
        )
        modularity = stractural_feature_calc.calc_modularity()
        return modularity.modularity

    def max_connectivity_between_layers(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ) -> float:
        between_layers_idx = get_between_layers_idx_from_feature_name(
            feature_name=feature_name,
        )
        out_connections = [
            self.out_degree(
                graph=graph,
                neuron=neuron,
                feature_name=feature_name,
            )
            for neuron in graph.layer_neuron_idx_mapping[between_layers_idx]
        ]
        return round(max(out_connections) / graph.dimensions[between_layers_idx + 1], 2)

    def total_connectivity_between_layers(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ) -> float:
        between_layers_idx = get_between_layers_idx_from_feature_name(
            feature_name=feature_name,
        )
        first_layer_dim, second_layer_dim = graph.dimensions[between_layers_idx:between_layers_idx + 2]
        max_possible_connections_between_layers = first_layer_dim * second_layer_dim
        connections_between_dims = sum(
            self.out_degree(
                graph=graph,
                neuron=neuron,
                feature_name=feature_name,
            )
            for neuron in graph.layer_neuron_idx_mapping[between_layers_idx]
        )
        connectivity = round(connections_between_dims / max_possible_connections_between_layers, 2)
        if self.connectivity_in_perc:
            return connectivity * 100
        return connectivity

    def num_paths_from_input_to_output(
            self,
            graph: Organism,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name_nx(
                feature_name=feature_name,
            )
        num_paths = 0
        for output_neuron in graph.layer_neuron_idx_mapping[-1]:
            if output_neuron not in graph.network.nodes(output_neuron):
                continue
            num_paths += sum(
                1
                for _ in nx.all_simple_paths(
                    graph.network,
                    source=neuron,
                    target=output_neuron,
                )
            )
        return num_paths

    def num_involved_neuron_in_paths_from_input_to_output(
            self,
            graph: Organism,
            feature_name: Optional[str],
            neuron: Optional[int] = None,
            *args,
    ) -> int:
        if neuron is None:
            neuron = get_neuron_number_from_feature_name_nx(
                feature_name=feature_name,
            )
        neuron_involved = set()
        for output_neuron in graph.layer_neuron_idx_mapping[-1]:
            if output_neuron not in graph.network.nodes(output_neuron):
                continue
            for path in nx.all_simple_paths(graph.network, source=neuron, target=output_neuron):
                neuron_involved = neuron_involved.union(set(path))
        return len(neuron_involved)

    def normed_entropy(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ):
        entropy = self.entropy(graph, feature_name)
        num_connections = len(graph.network.nodes)
        return entropy / np.log(
            min(num_connections, len(set(graph.network.nodes) - set(nx.isolates(graph.network)))))

    def motifs_count_0_(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ):
        g = nx.to_numpy_array(graph.network, weight=None, dtype=int)
        return self.motifs_count_0(
            graph=g,
            feature_name=feature_name,
        )

    def motifs_count_1_(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ):
        g = nx.to_numpy_array(graph.network, weight=None, dtype=int)
        return self.motifs_count_1(
            graph=g,
            feature_name=feature_name,
        )

    def motifs_count_2_(
            self,
            graph: Organism,
            feature_name: Optional[str],
            *args,
    ):
        g = nx.to_numpy_array(graph.network, weight=None, dtype=int)
        return self.motifs_count_2(
            graph=g,
            feature_name=feature_name,
        )

    def _motif_calculator(
            self,
            graph: Organism,
            motif_example: np.ndarray
    ):
        motif_example_graph = nx.from_numpy_array(motif_example, create_using=nx.DiGraph)
        matcher = isomorphism.DiGraphMatcher(graph.network, motif_example_graph)
        motifs = matcher.subgraph_isomorphisms_iter()
        return len(
            {
                frozenset(motif_mapping.keys())
                for motif_mapping in motifs
            }
        )

    def dimensions(
            self,
            graph,
            feature_name,
            *args,
    ):
        ind = get_between_layers_idx_from_feature_name(
            feature_name=feature_name,
        )
        return graph.dimensions[ind]
