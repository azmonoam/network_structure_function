import random
from typing import Dict, List

import numpy as np

from new_organism import Organism
from new_reproduction import NReproduction
from utils.set_up_organism_utils import NodeType


class NReproductionRandomDim(NReproduction):

    def __init__(
            self,
            mutation_probabilities: Dict,
            first_parent: Organism,
            second_parent: Organism,
    ):
        super().__init__(mutation_probabilities, first_parent, second_parent)

    def _get_neurons_and_assign_biases_from_parents(self) -> List[List[NodeType]]:
        neurons_from_parents = [[], []]
        parents = [self.first_parent, self.second_parent]
        for neuron in self.leading_parent.network.nodes:
            random_order_parents_ind = random.sample([0, 1], 2)
            if parents[random_order_parents_ind[0]].network.has_node(neuron):
                i = random_order_parents_ind[0]
            else:
                i = random_order_parents_ind[1]
            neurons_from_parents[i].append(neuron)
        return neurons_from_parents

    def _init_org_properties(self):
        self.leading_parent = random.choice([self.first_parent, self.second_parent])
        self.num_layers = self.leading_parent.num_layers
        self.allowed_bias_values = self.leading_parent.allowed_bias_values
        self.allowed_weights_values = self.leading_parent.allowed_weights_values
        self.use_bias = self.leading_parent.use_bias
        self.communities_input_symmetry = self.leading_parent.communities_input_symmetry
        # ToDO: if biases are relvent then need to implement a way to chose them from the parant of the chosen neuron
        self.bias_list = self.leading_parent.biases
        random_connectivity_ratio = random.choice(
            [
                self.first_parent.network.number_of_edges() / len(self.first_parent.all_allowed_edges),
                self.second_parent.network.number_of_edges() / len(self.second_parent.all_allowed_edges),
            ]
        )
        self.max_allowed_connections = int(
            np.floor(len(self.leading_parent.all_allowed_edges) * random_connectivity_ratio)
        )

    def _add_extra_connection(
            self,
            parents: List[Organism],
            neurons_from_parents: List[List[NodeType]],
            child_neurons: List[NodeType],
    ):
        existing_edges_no_weight = [
            (existing_edge[0], existing_edge[1])
            for existing_edge in self.edges
        ]
        i, j = random.sample([0, 1], 2)
        random_parent_unused_nodes_edges = list(parents[j].network.out_edges(neurons_from_parents[i], data=True))
        edge_added = self._check_possible_edges_and_add_if_possible(
            random_parent_unused_nodes_edges=random_parent_unused_nodes_edges,
            existing_edges_no_weight=existing_edges_no_weight,
            child_neurons=child_neurons,
        )
        if edge_added:
            return
        random_parent_unused_nodes_edges = list(parents[i].network.out_edges(neurons_from_parents[j], data=True))
        edge_added = self._check_possible_edges_and_add_if_possible(
            random_parent_unused_nodes_edges=random_parent_unused_nodes_edges,
            existing_edges_no_weight=existing_edges_no_weight,
            child_neurons=child_neurons,
        )
        if edge_added:
            return
        all_allowed_edges = set(self.leading_parent.all_allowed_edges)
        optional_edges_to_add = list(all_allowed_edges - set(existing_edges_no_weight))
        u, v = random.choice(optional_edges_to_add)
        self.edges.append(
            (u, v, {'weight': np.random.choice(self.allowed_weights_values)})
        )

    def _check_possible_edges_and_add_if_possible(
            self,
            random_parent_unused_nodes_edges: List[NodeType],
            existing_edges_no_weight: List[NodeType],
            child_neurons: List[NodeType],
    ) -> bool:
        random.shuffle(random_parent_unused_nodes_edges)
        for edge in random_parent_unused_nodes_edges:
            if edge[0] in child_neurons and edge[1] in child_neurons:
                if (edge[0], edge[1]) not in existing_edges_no_weight:
                    self.edges.append(edge)
                    return True
        return False
