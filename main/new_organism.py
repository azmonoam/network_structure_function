import random
from itertools import chain
from typing import List, Optional, Union, Tuple
from utils.main_utils import clean_graph_from_unused_nodes
import copy
import networkx as nx
import numpy as np

from stractural_features_models.structural_features import StructuralFeatures
from utils.set_up_organism_utils import get_allowed_edges, get_tuple_layer_neuron_idx_mapping, \
    get_number_of_neurons_per_layer, NodeType, EdgeType


class Organism:

    def __init__(
            self,
            dimensions: List[int],
            num_layers: int,
            allowed_weights_values: List[int],
            allowed_bias_values: Optional[List[int]] = None,
            connection_cost: Optional[float] = None,
            predictions: Optional[List[np.ndarray]] = None,
            performance: Optional[float] = None,
            fitness: Optional[Tuple[float]] = None,
            generation: Optional[int] = 0,
            network: Optional[nx.DiGraph] = nx.DiGraph(),
            clean_network: Optional[nx.DiGraph] = None,
            biases: Optional[List[np.ndarray]] = list(),
            use_bias: bool = True,
            layer_neuron_idx_mapping: Optional[List[List[Tuple[int, int]]]] = None,
            max_allowed_connections: int = None,
            communities_input_symmetry: int = 2,
            structural_features: Optional[StructuralFeatures] = None,
    ):
        self.dimensions = dimensions
        self.connection_cost = connection_cost
        self.predictions = predictions
        self.performance = performance
        self.fitness = fitness
        self.generation = generation
        self.network = network
        self.clean_network = clean_network
        self.biases = biases
        self.use_bias = use_bias
        self.structural_features = structural_features
        if not self.structural_features:
            self.structural_features = StructuralFeatures()
        self.allowed_weights_values = allowed_weights_values
        self.num_layers = num_layers
        self.max_allowed_connections = max_allowed_connections
        self.communities_input_symmetry = communities_input_symmetry
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
        self.allowed_bias_values = self._get_allowed_bias_values(
            allowed_bias_values=allowed_bias_values
        )

    def build_organism(
            self,
            max_connections_per_neuron_in_layer: Optional[List[int]] = None,
    ):
        nodes = []
        edges = []
        if max_connections_per_neuron_in_layer:
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
            max_allowed_connections: int,
    ):
        self.max_allowed_connections = max_allowed_connections
        weights = np.random.choice(self.allowed_weights_values, (max_allowed_connections,))
        self.network.add_nodes_from(chain(*self.layer_neuron_idx_mapping))
        edges = self._choose_edges(weights=weights)
        self.network.add_edges_from(edges)
        for node, degree in self.network.in_degree():
            if degree == 0 and node[0] > 0:
                self._switch_edge(
                    node=node,
                    in_or_out='in',
                )
        for node, degree in self.network.out_degree():
            if degree == 0 and node[0] < self.num_layers:
                self._switch_edge(
                    node=node,
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
            non_input_output_edges = [
                edge
                for edge in shuffled_edges if
                (edge[0] not in self.layer_neuron_idx_mapping[0] and edge[1] not in self.layer_neuron_idx_mapping[-1])
            ]
            self.network.remove_edge(*random.choice(non_input_output_edges))
            print('warning: there where no edges with degree higher then 1 - removed a random connection. '
                  f'Mey be an issue with the connectivity ratio')

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
        if set(nx.isolates(self.network)):
            print('Note - there are some unconnected nodes!')

    def _choose_edges_(
            self,
            weights
    ):
        free_edges = list(self.all_allowed_edges)
        edges = []
        i = 0
        for input_neuron in self.layer_neuron_idx_mapping[0]:
            edge = (input_neuron, random.choice(self.layer_neuron_idx_mapping[1]))
            free_edges.pop(free_edges.index(edge))
            edges.append((edge[0], edge[1], {'weight': weights[i]}))
            i += 1
        for output_neuron in self.layer_neuron_idx_mapping[-1]:
            edge = (random.choice(self.layer_neuron_idx_mapping[-2]), output_neuron)
            free_edges.pop(free_edges.index(edge))
            edges.append((edge[0], edge[1], {'weight': weights[i]}))
            i += 1
        return edges + [
            (output_neuron, input_neuron, {'weight': weights[idx + i]})
            for idx, (output_neuron, input_neuron) in
            enumerate(random.sample(free_edges, self.max_allowed_connections - i))
        ]

    def _switch_edge(
            self,
            node: NodeType,
            in_or_out: str,
    ):
        other_nodes_in_layer = copy.deepcopy(self.layer_neuron_idx_mapping[node[0]])
        random.shuffle(other_nodes_in_layer)
        for other_node in other_nodes_in_layer:
            if in_or_out == 'in':
                if self.network.in_degree(other_node) > 1:
                    edge_to_switch_ind = random.randint(0, len(self.network.in_edges(other_node))-1)
                    out_node, _ = list(self.network.in_edges(other_node))[edge_to_switch_ind]
                    self.network.remove_edge(out_node, other_node)
                    self.network.add_edge(
                        out_node, node, weight=random.choice(self.allowed_weights_values)
                    )
                    return
            else:
                if self.network.out_degree(other_node) > 1:
                    edge_to_switch_ind = random.randint(0, len(self.network.out_edges(other_node)-1))
                    _, in_node = list(self.network.out_edges(other_node))[edge_to_switch_ind]
                    self.network.remove_edge(other_node, in_node)
                    self.network.add_edge(
                        node, in_node, weight=random.choice(self.allowed_weights_values)
                    )
                    return
        shuffled_edges = list(self.network.edges())
        random.shuffle(shuffled_edges)
        for out_node, in_node in shuffled_edges:
            if self.network.out_degree(out_node) > 1 and self.network.in_degree(in_node) > 1:
                self.network.remove_edge(out_node, in_node)
                self._add_edge_for_neuron(
                    neuron=node,
                    in_or_out=in_or_out,
                )
                return

    def _choose_edges(
            self,
            weights: np.ndarray,
    ):
        edges = []
        free_edges = list(self.all_allowed_edges)
        for layer_ind in range(self.num_layers):
            neurons_without_in_connection = copy.deepcopy(self.layer_neuron_idx_mapping[layer_ind+1])
            for i, output_neuron in enumerate(random.sample(self.layer_neuron_idx_mapping[layer_ind], len(self.layer_neuron_idx_mapping[layer_ind]))):
                if layer_ind + i > self.max_allowed_connections:
                    raise ValueError('Not enough edges to give!')
                if neurons_without_in_connection:
                    input_neuron = random.choice(neurons_without_in_connection)
                    neurons_without_in_connection.remove(input_neuron)
                else:
                    input_neuron = random.choice(self.layer_neuron_idx_mapping[layer_ind + 1])
                edges.append((output_neuron, input_neuron, {'weight': weights[layer_ind + i]}))
                free_edges.remove((output_neuron, input_neuron))
        return edges + [
            (output_neuron, input_neuron, {'weight': weights[idx + len(edges)]})
            for idx, (output_neuron, input_neuron) in
            enumerate(random.sample(free_edges, self.max_allowed_connections - len(edges)))
        ]
