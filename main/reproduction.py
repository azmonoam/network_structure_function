import random
from itertools import chain
from typing import List, Dict

import networkx as nx
import numpy as np

from neuroevolution import Neuroevolution
from organism import Organism
from utils.set_up_organism_utils import get_allowed_edges, NodeType, get_tuple_layer_neuron_idx_mapping


class Reproduction:

    def __init__(
            self,
            neuroevolution: Neuroevolution,
            first_parent: Organism,
            second_parent: Organism,
    ):
        self.first_parent = first_parent
        self.second_parent = second_parent
        self.bias_mutation_probability = neuroevolution.mutation_probabilities.get('bias_mutation_probability')
        self.connection_switch_mutation_probability = neuroevolution.mutation_probabilities.get(
            'connection_switch_mutation_probability')
        self.connection_addition_mutation_probability = neuroevolution.mutation_probabilities.get(
            'connection_addition_mutation_probability')
        self.connection_removal_mutation_probability = neuroevolution.mutation_probabilities.get(
            'connection_removal_mutation_probability')
        self.weight_mutation_probability_factor = neuroevolution.mutation_probabilities.get(
            'weight_mutation_probability_factor')
        self.neuron_addition_mutation_probability = neuroevolution.mutation_probabilities.get(
            'neuron_addition_mutation_probability')
        self.neuron_removal_mutation_probability = neuroevolution.mutation_probabilities.get(
            'neuron_removal_mutation_probability')
        self.neuroevolution = neuroevolution
        self.num_layers = neuroevolution.num_layers
        self.allowed_bias_values = self.first_parent.allowed_bias_values
        self.allowed_weights_values = self.first_parent.allowed_weights_values
        self.use_bias = self.first_parent.use_bias
        self.max_allowed_connections = self.first_parent.max_allowed_connections
        self.bias_list = [[] for _ in range(len(self.first_parent.biases))]
        self.layer_neuron_idx_mapping = [[] for _ in range(self.num_layers + 1)]
        self.edges = []

    def mate(self) -> Organism:
        self._check_values()
        neurons_from_parents = [[], []]
        parents = [self.first_parent, self.second_parent]
        for neuron in sorted(set(self.first_parent.network.nodes) | set(self.second_parent.network.nodes)):
            layer, neuron_idx = neuron
            i = random.choice([0, 1])
            if parents[i].network.has_node(neuron):
                neurons_from_parents[i].append(neuron)
                if parents[i].use_bias and layer > 0:
                    self.bias_list[layer - 1].append(parents[i].biases[layer - 1][0, neuron_idx])
        child_neurons = sorted(chain(*neurons_from_parents))
        self._add_parents_edges(
            neurons_from_parents=neurons_from_parents,
            child_neurons=child_neurons,
        )
        self.layer_neuron_idx_mapping = self._get_layer_neuron_idx_mapping(
            child_neurons=child_neurons,
        )
        self._test_number_of_child_edges()
        self._mutate()
        dimensions = [
            len(layer_neurons)
            for layer_neurons in self.layer_neuron_idx_mapping
        ]
        self._rename_child_properties(
            dimensions=dimensions,
        )
        self._test_number_of_child_edges()
        network = nx.DiGraph()
        network.add_nodes_from(chain(*self.layer_neuron_idx_mapping))
        network.add_edges_from(self.edges)
        return Organism(
            dimensions=dimensions,
            layer_neuron_idx_mapping=self.layer_neuron_idx_mapping,
            network=network,
            biases=self._rearrange_biases(),
            use_bias=self.use_bias,
            neuroevolution=self.neuroevolution,
            max_allowed_connections=self.max_allowed_connections
        )

    def _mutate(self):
        if self.neuron_removal_mutation_probability is not None:
            if random.uniform(0, 1) < self.neuron_removal_mutation_probability \
                    and not all(
                len(layer) <= 1
                for layer in self.layer_neuron_idx_mapping[1:-1]
            ):
                self._remove_neuron()
        if self.neuron_addition_mutation_probability is not None:
            if random.uniform(0, 1) < self.neuron_addition_mutation_probability:
                self._add_neuron()
        if self.connection_switch_mutation_probability is not None:
            if random.uniform(0, 1) < self.connection_switch_mutation_probability:
                self._switch_connection()
        if self.connection_removal_mutation_probability is not None:
            if random.uniform(0, 1) < self.connection_removal_mutation_probability and len(self.edges) > 1:
                self._remove_connection()
        if self.connection_addition_mutation_probability is not None:
            if random.uniform(0, 1) < self.connection_addition_mutation_probability:
                self._add_connection()
        if self.weight_mutation_probability_factor is not None:
            weight_mutation_probability = self.weight_mutation_probability_factor / len(self.edges)
            for idx, edge in enumerate(self.edges):
                if random.uniform(0, 1) < weight_mutation_probability:
                    self._change_weight_value(idx=idx)
        if self.use_bias and self.bias_mutation_probability is not None:
            for layer_idx, layer_biases in enumerate(self.bias_list):
                for neuron in range(len(layer_biases)):
                    if random.uniform(0, 1) < self.bias_mutation_probability:
                        self._change_bias_value(
                            layer_idx=layer_idx,
                            neuron=neuron,
                        )

    def _check_values(self):
        if self.first_parent.use_bias != self.second_parent.use_bias:
            raise ValueError('Both parents must use bias or not use bias')
        if self.first_parent.allowed_weights_values != self.second_parent.allowed_weights_values:
            raise ValueError('Both parents must have the same allowed weight values')
        if self.first_parent.allowed_bias_values != self.second_parent.allowed_bias_values:
            raise ValueError('Both parents must have the same allowed bias values')

    def _rename_child_properties(
            self,
            dimensions: List[int],
    ):
        edges_to_rename = {}
        correct_layer_neuron_idx_mapping = get_tuple_layer_neuron_idx_mapping(
            start_dimensions=dimensions,
        )
        for layer in range(self.num_layers + 1):
            for correct_layers_neuron, layer_neuron in zip(
                    correct_layer_neuron_idx_mapping[layer], self.layer_neuron_idx_mapping[layer]
            ):
                if correct_layers_neuron != layer_neuron:
                    edges_to_rename[layer_neuron] = correct_layers_neuron
        self._rename_edges(
            edges_to_rename=edges_to_rename,
        )
        self.layer_neuron_idx_mapping = correct_layer_neuron_idx_mapping

    def _rename_edges(
            self,
            edges_to_rename: Dict[NodeType, NodeType]
    ):
        updated_edges = []
        for out_neuron, in_neuron, data in self.edges:
            if out_neuron in edges_to_rename.keys():
                out_neuron = edges_to_rename[out_neuron]
            if in_neuron in edges_to_rename.keys():
                in_neuron = edges_to_rename[in_neuron]
            updated_edges.append((out_neuron, in_neuron, data))
        self.edges = updated_edges

    def _add_parents_edges(
            self,
            neurons_from_parents: List[List[NodeType]],
            child_neurons: List[NodeType],
    ):
        parents = [self.first_parent, self.second_parent]
        for parent_num, parent in enumerate(parents):
            for edge in parent.network.out_edges(neurons_from_parents[parent_num], data=True):
                if len(self.edges) >= self.max_allowed_connections:
                    break
                if edge[1] in child_neurons:
                    self.edges.append(edge)
        while len(self.edges) < self.max_allowed_connections:
            self._add_extra_connection(
                parents=parents,
                neurons_from_parents=neurons_from_parents,
                child_neurons=child_neurons,
            )

    def _add_extra_connection(
            self,
            parents: List[Organism],
            neurons_from_parents: List[List[NodeType]],
            child_neurons: List[NodeType],
    ):
        i = random.choice([0, 1])
        j = 0 if i == 1 else 1
        random_parent_unused_nodes_edges = list(parents[j].network.out_edges(neurons_from_parents[i], data=True))
        existing_edges_no_weight = [
            (existing_edge[0], existing_edge[1])
            for existing_edge in self.edges
        ]
        edge = random.choice(random_parent_unused_nodes_edges)
        if edge[0] in child_neurons and edge[1] in child_neurons:
            if (edge[0], edge[1]) not in existing_edges_no_weight:
                self.edges.append(edge)

    def _get_layer_neuron_idx_mapping(
            self,
            child_neurons: List[NodeType],
    ) -> List[List[NodeType]]:
        for neuron_name in child_neurons:
            layer = neuron_name[0]
            self.layer_neuron_idx_mapping[layer].append(neuron_name)
        return self.layer_neuron_idx_mapping

    def _rearrange_biases(
            self,
    ) -> List[np.ndarray]:
        return [
            np.array(layers_bias_vals).reshape(1, -1)
            for layers_bias_vals in self.bias_list
        ]

    def _get_random_layer(self) -> int:
        return random.randint(1, len(self.layer_neuron_idx_mapping) - 2)

    def _remove_neuron(self):
        layer = self._get_random_layer()
        while len(self.layer_neuron_idx_mapping[layer]) <= 1:
            layer = random.randint(1, len(self.layer_neuron_idx_mapping) - 2)
        neuron_to_remove = random.choice(self.layer_neuron_idx_mapping[layer])
        updated_edges = [
            edge
            for edge in self.edges
            if neuron_to_remove not in edge
        ]
        self.edges = updated_edges
        neuron_to_remove_idx = self.layer_neuron_idx_mapping[layer].index(neuron_to_remove)
        self.bias_list[layer - 1].pop(neuron_to_remove_idx)
        self.layer_neuron_idx_mapping[layer].pop(neuron_to_remove_idx)

    def _add_neuron(self):
        layer = self._get_random_layer()
        neuron = max(self.layer_neuron_idx_mapping[layer])[1] + 1
        node_to_add = (layer, neuron)
        self.layer_neuron_idx_mapping[layer].append(node_to_add)
        self.bias_list[layer - 1].append(random.choice(self.allowed_bias_values))

    def _remove_connection(self):
        edge_to_remove = random.randint(0, len(self.edges) - 1)
        self.edges.pop(edge_to_remove)

    def _add_connection(self):
        all_allowed_edges = set(chain(*get_allowed_edges(self.layer_neuron_idx_mapping)))
        existing_edges = {(u, v) for u, v, _ in self.edges}
        optional_edges_to_add = list(all_allowed_edges - existing_edges)
        if optional_edges_to_add:
            u, v = random.choice(optional_edges_to_add)
            self.edges.append(
                (u, v, {'weight': np.random.choice(self.allowed_weights_values)})
            )

    def _change_weight_value(
            self,
            idx: int
    ):
        self.edges[idx][2]['weight'] = np.random.choice(self.allowed_weights_values)

    def _change_bias_value(
            self,
            layer_idx: int,
            neuron: int,
    ):
        self.bias_list[layer_idx][neuron] = random.choice(self.allowed_bias_values)

    def _switch_connection(
            self,
    ):
        self._remove_connection()
        self._add_connection()

    def _test_number_of_child_edges(self):
        child_num_edge = len(self.edges)
        first_parent_num_edges = self.first_parent.network.number_of_edges()
        second_parent_num_edges = self.second_parent.network.number_of_edges()
        if child_num_edge != first_parent_num_edges or child_num_edge != second_parent_num_edges:
            raise Exception(f'child number of edges ({child_num_edge}) is different then his parents'
                            f' ({first_parent_num_edges},{second_parent_num_edges})')
