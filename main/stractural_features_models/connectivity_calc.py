from itertools import combinations
from typing import List, Dict, Union, Tuple

import networkx as nx
import numpy as np

from new_organism import Organism
from stractural_features_models.structural_features import ConnectivityFeatures
from utils.main_utils import get_number_of_connections


class ConnectivityCalc:
    def __init__(
            self,
            dimensions: List[int],
            num_layers: int,
    ):
        self.dimensions = dimensions
        self.num_layers = num_layers
        self.CONNECTIVITY_DATA_KEYS = [
            'out_connections_per_layer',
            'in_connections_per_layer',
            'total_connectivity_ratio_between_layers',
            'max_connectivity_between_layers_per_layer',
            'mean_connectivity_between_layers_per_layer',
        ]

    def get_connectivity(
            self,
            organism: Organism,
    ):
        max_possible_connections = self.calc_max_possible_connections()
        num_connections = get_number_of_connections(
            organism=organism,
        )
        connectivity_ratio = round(num_connections / max_possible_connections, 2)
        num_paths_to_output, num_involved_neurons_in_paths = self._calc_connectivity_between_input_and_output(
            organism=organism,
        )
        return ConnectivityFeatures(
            max_possible_connections=max_possible_connections,
            num_connections=num_connections,
            connectivity_ratio=connectivity_ratio,
            connectivity_data_dict_per_layer=self._calc_connectivity_data_per_layer(
                organism=organism,
            ),
            distances_between_input_neuron=self._calc_distance_between_input_neurons(
                organism=organism,
            ),
            num_paths_to_output_per_input_neuron=num_paths_to_output,
            num_involved_neurons_in_paths_per_input_neuron=num_involved_neurons_in_paths,
        )

    def calc_max_possible_connections(self) -> int:
        return sum(
            self.dimensions[i] * self.dimensions[i + 1]
            for i in range(self.num_layers)
        )

    def _calc_connectivity_data_per_layer(
            self,
            organism: Organism,
    ) -> Dict[str, List[Union[float, int]]]:
        connectivity_data = {
            key: []
            for key in self.CONNECTIVITY_DATA_KEYS
        }
        for layer in range(self.num_layers + 1):
            num_out_connections_in_layer = [
                organism.clean_network.out_degree(neuron)
                if neuron in organism.clean_network.nodes(neuron) else 0
                for neuron in organism.layer_neuron_idx_mapping[layer]
            ]
            num_in_connections_in_layer = [
                organism.clean_network.in_degree(neuron)
                if neuron in organism.clean_network.nodes(neuron) else 0
                for neuron in organism.layer_neuron_idx_mapping[layer]

            ]
            connectivity_data['out_connections_per_layer'].append(num_out_connections_in_layer)
            connectivity_data['in_connections_per_layer'].append(num_in_connections_in_layer)
            if layer != self.num_layers:
                neurons_in_next_layer = self.dimensions[layer + 1]
                max_possible_connections_from_layer = self.dimensions[layer] * neurons_in_next_layer
                connectivity_data['total_connectivity_ratio_between_layers'].append(
                    round(sum(num_out_connections_in_layer) / max_possible_connections_from_layer, 2)
                )
                connectivity_data['max_connectivity_between_layers_per_layer'].append(
                    round(max(num_out_connections_in_layer) / neurons_in_next_layer, 2)
                )
        connectivity_data['layer_connectivity_rank'] = np.argsort(
            connectivity_data['total_connectivity_ratio_between_layers'])[::-1]
        return connectivity_data

    @staticmethod
    def _calc_connectivity_between_input_and_output(
            organism: Organism,
    ) -> Tuple[List[int], List[int]]:
        num_paths_to_output = []
        num_involved_neurons_in_paths = []
        for input_neuron in organism.layer_neuron_idx_mapping[0]:
            num_paths = 0
            neuron_involved = set()
            if input_neuron not in organism.clean_network.nodes(input_neuron):
                num_paths_to_output.append(num_paths)
                num_involved_neurons_in_paths.append(len(neuron_involved))
                continue
            for output_neuron in organism.layer_neuron_idx_mapping[-1]:
                if output_neuron not in organism.clean_network.nodes(output_neuron):
                    continue
                for path in nx.all_simple_paths(organism.clean_network, source=input_neuron, target=output_neuron):
                    num_paths += 1
                    neuron_involved = neuron_involved.union(set(path))
            num_paths_to_output.append(num_paths)
            num_involved_neurons_in_paths.append(len(neuron_involved))
        return num_paths_to_output, num_involved_neurons_in_paths

    @staticmethod
    def _calc_distance_between_input_neurons(
            organism: Organism,
    ) -> List[int]:
        distance_between_input_neurons = []
        undirected_org = organism.clean_network.to_undirected()
        for first_input_neuron, second_input_neuron in combinations(organism.layer_neuron_idx_mapping[0], 2):
            if first_input_neuron not in organism.clean_network.nodes(first_input_neuron) \
                    or second_input_neuron not in organism.clean_network.nodes(second_input_neuron):
                distance = -1
                distance_between_input_neurons.append(distance)
                continue
            try:
                distance = len(
                    nx.shortest_path(
                        undirected_org,
                        source=first_input_neuron,
                        target=second_input_neuron,
                    )
                ) / 2
            except nx.exception.NetworkXNoPath:
                distance = -1
            distance_between_input_neurons.append(distance)
        return distance_between_input_neurons
