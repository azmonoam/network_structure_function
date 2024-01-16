import random
from itertools import chain
from typing import List, Optional, Union, Tuple

import networkx as nx
import numpy as np

from neuroevolution import Neuroevolution
from utils.set_up_organism_utils import get_allowed_edges, get_tuple_layer_neuron_idx_mapping, \
    get_number_of_neurons_per_layer, NodeType, EdgeType


class Organism:

    def __init__(
            self,
            neuroevolution: Neuroevolution,
            dimensions: List[int],
            biases: Optional[List[np.ndarray]] = None,
            network: Optional[nx.DiGraph] = None,
            layer_neuron_idx_mapping: Optional[List[List[Tuple[int, int]]]] = None,
            connectivity_ratio: float = None,
            max_allowed_connections: int = None,
            use_bias: bool = True,
    ):
        self.dimensions = dimensions
        self.connection_cost = None
        self.predictions = None
        self.performance = None
        self.fitness = None
        self.modularity = None
        self.communities = None
        self.generation = None
        if network:
            self.network = network
        else:
            self.network = nx.DiGraph()
        self.clean_network = None
        if biases:
            self.biases = biases
        else:
            self.biases = []
        self.use_bias = use_bias
        if layer_neuron_idx_mapping:
            self.layer_neuron_idx_mapping = layer_neuron_idx_mapping
        else:
            self.layer_neuron_idx_mapping = get_tuple_layer_neuron_idx_mapping(
                start_dimensions=dimensions,
            )
        self.layer_neuron_indexes = get_number_of_neurons_per_layer(
            layer_neuron_idx_mapping=self.layer_neuron_idx_mapping,
        )
        self.allowed_edges = get_allowed_edges(
            layer_neuron_idx_mapping=self.layer_neuron_idx_mapping,
        )
        self.all_allowed_edges = set(chain(*self.allowed_edges))
        self.allowed_weights_values = neuroevolution.allowed_weights_values
        self.allowed_bias_values = self._get_allowed_bias_values(
            allowed_bias_values=neuroevolution.allowed_bias_values
        )
        self.max_allowed_connections = max_allowed_connections
        if connectivity_ratio:
            self.connectivity_ratio = connectivity_ratio
        else:
            self.connectivity_ratio = None
        self.num_layers = neuroevolution.num_layers

    def build_organism(
            self,
            max_connections_per_neuron_in_layer: Optional[List[int]] = None,
    ):
        nodes = []
        edges = []
        if len(max_connections_per_neuron_in_layer) != self.num_layers:
            raise ValueError('max connections per layers have to be of length layers + 1')
        weights = self._get_weights_array(
            max_connections_per_neuron_in_layer=max_connections_per_neuron_in_layer,
        )
        for i in range(self.num_layers):
            current_layer_neurons, next_layer_neurons = (
                self.layer_neuron_idx_mapping[i], self.layer_neuron_idx_mapping[i + 1])
            nodes += [node for node in current_layer_neurons]
            allowed_edges = self._get_allowed_edges(
                layer=i,
                max_connections_per_neuron_in_layer=max_connections_per_neuron_in_layer,
                current_layer_neurons=current_layer_neurons,
                next_layer_neurons=next_layer_neurons,
            )
            for out_neuron, in_neuron in allowed_edges:
                edges.append(
                    (out_neuron, in_neuron, {'weight': weights[len(edges)]})
                )
            self._add_biases_for_layer(layer=i)
        nodes += [node for node in next_layer_neurons]
        self.network.add_nodes_from(nodes)
        self.network.add_edges_from(edges)

    def build_organism_by_connectivity(
            self,
            max_allowed_connections: int
    ):
        self.max_allowed_connections = max_allowed_connections
        weights = np.random.choice(self.allowed_weights_values, (max_allowed_connections,))
        self.network.add_nodes_from(chain(*self.layer_neuron_idx_mapping))
        edges = [
            (out_neuron, in_neuron, {'weight': weights[idx]})
            for idx, (out_neuron, in_neuron) in
            enumerate(random.sample(list(self.all_allowed_edges), max_allowed_connections))]
        self.network.add_edges_from(edges)
        for node, degree in self.network.in_degree():
            if degree == 0 and node[0] > 0:
                self._add_edge_for_neuron(
                    neuron=node,
                    in_or_out='in',
                )
                self._remove_edge_from_node_with_high_degree()
        for node, degree in self.network.out_degree():
            if degree == 0 and node[0] < self.num_layers:
                self._add_edge_for_neuron(
                    neuron=node,
                    in_or_out='out',
                )
                self._remove_edge_from_node_with_high_degree()
        for layer in range(self.num_layers):
            self._add_biases_for_layer(layer=layer)
        self._test_number_of_edges()

    def predict(
            self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        if not input_array.ndim == 2:
            raise ValueError(f'Input has {input_array.ndim} dimensions, expected 2')
        layers = self.convert_network_to_list_of_layers_arrays()
        for index, (layer, bias) in enumerate(zip(layers, self.biases)):
            input_array = input_array @ layer + np.ones((input_array.shape[0], 1)) @ bias
            if index == len(layers) - 1:
                input_array = self._softmax(input_array)  # output activation
            else:
                input_array = np.tanh(20 * input_array)
        return input_array

    @staticmethod
    def _softmax(
            x: np.ndarray,
    ) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def convert_network_to_list_of_layers_arrays(
            self,
            weight: Optional[str] = 'weight',
            datatype: Optional[np.dtype] = int,
    ) -> List[np.ndarray]:
        network_array = nx.to_numpy_array(self.network, weight=weight, dtype=datatype)
        return [
            network_array[rows[0]:rows[1], columns[0]: columns[1]]
            for rows, columns in self.layer_neuron_indexes
        ]

    @staticmethod
    def _get_allowed_bias_values(
            allowed_bias_values: Optional[List[int]]
    ) -> List[int]:
        return allowed_bias_values if allowed_bias_values is not None else [0]

    def _get_max_connections(
            self,
            max_connections_per_neuron_in_layer: Optional[List[int]],
    ):
        if max_connections_per_neuron_in_layer:
            return sum(self.dimensions[layer] * max_connections for layer, max_connections in
                       enumerate(max_connections_per_neuron_in_layer))
        return sum(self.dimensions[i] * self.dimensions[i + 1] for i in range(self.num_layers))

    def _get_weights_array(
            self,
            max_connections_per_neuron_in_layer: Optional[List[int]],
    ) -> np.ndarray:
        max_connections = self._get_max_connections(
            max_connections_per_neuron_in_layer=max_connections_per_neuron_in_layer,
        )
        return np.random.choice(self.allowed_weights_values, (max_connections,))

    def _get_allowed_edges(
            self,
            layer: int,
            max_connections_per_neuron_in_layer: Optional[List[Union[int, str]]],
            current_layer_neurons: List[Union[int, Tuple[int, int]]],
            next_layer_neurons: List[Union[int, Tuple[int, int]]],
    ) -> List[EdgeType]:
        if max_connections_per_neuron_in_layer:
            allowed_edges = []
            for out_neuron in current_layer_neurons:
                in_neurons = random.sample(next_layer_neurons, max_connections_per_neuron_in_layer[layer])
                allowed_edges += [(out_neuron, in_neuron) for in_neuron in in_neurons]
            return allowed_edges
        return self.allowed_edges[layer]

    def _add_edge_for_neuron(
            self,
            neuron: NodeType,
            in_or_out: str,
    ):
        neurons_layer_idx = neuron[0]
        in_or_out_idx = 0
        if in_or_out == 'in':
            in_or_out_idx = 1
            neurons_layer_idx -= 1
        allowed_edges_for_neuron = [
            edge
            for edge in self.allowed_edges[neurons_layer_idx]
            if edge[in_or_out_idx] == neuron
        ]
        u, v = random.choice(allowed_edges_for_neuron)
        self.network.add_edge(
            u, v, weight=random.choice(self.allowed_weights_values)
        )

    def _remove_edge_from_node_with_high_degree(self):
        removed = False
        shuffled_edges = list(self.network.edges())
        for out_node, in_node in shuffled_edges:
            if self.network.out_degree(out_node) > 1 and self.network.in_degree(in_node) > 1:
                self.network.remove_edge(out_node, in_node)
                removed = True
                break
        if not removed:
            self.network.remove_edge(*random.choice(shuffled_edges))
            print('warning: there where no edges with degree higher then 1 - removed a random connection. '
                  f'Mey be an issue with the connectivity ratio ({self.connectivity_ratio})')

    def _add_biases_for_layer(
            self,
            layer: int,
    ):
        bias = np.random.choice(self.allowed_bias_values, (1, self.dimensions[layer + 1])) * self.use_bias
        self.biases.append(bias)

    def _test_number_of_edges(self):
        num_edges = self.network.number_of_edges()
        if num_edges != self.max_allowed_connections:
            raise Exception(f'number of edges ({num_edges}) is different then max_allowed_connections'
                            f' ({self.max_allowed_connections})')
