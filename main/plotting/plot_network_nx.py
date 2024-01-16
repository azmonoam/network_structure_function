import copy
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx

from organism import Organism as OldOrg
from new_organism import Organism
from utils.main_utils import COLORS


def draw_network(
        organism: OldOrg,
        generation: int,
        network_file_name: str,
        title: bool,
):
    output_nudes = organism.layer_neuron_idx_mapping[-1]
    organism.clean_network.remove_nodes_from(output_nudes)
    neurons_mapping = _get_neurons_to_layer_and_community_mapping(
        network=organism.clean_network,
        dimensions=organism.dimensions,
    )
    coordinates = _get_coordinates(
        neurons_mapping
    )
    _draw(
        network=organism.clean_network,
        coordinates=coordinates,
        ind=generation,
        fitness=organism.performance,
        file_name=network_file_name,
        title=title,
    )


def _draw(
        network: nx.DiGraph,
        coordinates: Dict[int, Tuple[float, float]],
        ind: int,
        fitness: float,
        file_name: Optional[str] = None,
        title: Optional[bool] = True,
        with_labels: bool = False,
):
    plt.figure(figsize=(5, 6), dpi=80)
    if title:
        plt.title(f'Generation: {ind}, fitness: {fitness:.2f}')
    widths = nx.get_edge_attributes(network, 'weight')
    line_width = [x / 2 for x in list(widths.values())]
    node_color = [COLORS[network.nodes[v]['community']] for v in network.nodes]
    nx.draw(network, coordinates, with_labels=with_labels, arrows=False, edgelist=widths.keys(),
            width=line_width, node_color=node_color, )
    if file_name:
        plt.savefig(file_name, dpi=80)
    plt.close('all')


def _get_neurons_to_layer_and_community_mapping(
        network: nx.DiGraph,
        dimensions: List[int],
        num_communities: int = 2,
) -> Dict[int, List[List[int]]]:
    list_template = [
        [] for _ in range(num_communities)
    ]
    neurons_mapping = {
        layer: copy.deepcopy(list_template) for layer in range(len(dimensions) - 1)
    }
    for neuron in network.nodes():
        layer = neuron[0]
        if layer in neurons_mapping.keys():
            community = network.nodes[neuron]['community']
            neurons_mapping[layer][community].append(neuron)
    return neurons_mapping


def _get_coordinates(
        neurons_mapping: Dict[int, List[List[int]]],
        min_x: float = 30,
        max_x: float = 600,
        min_y: float = 20,
        max_y: float = 300,
) -> Dict[int, Tuple[float, float]]:
    coordinates = {}
    num_neurons_in_layers = [
        sum(len(x) for x in neurons_list) for neurons_list in neurons_mapping.values()
    ]
    number_of_neurons_in_widest_layer = num_neurons_in_layers[0]
    vertical_distance_between_layers = (max_y - min_y) / len(num_neurons_in_layers)
    horizontal_distance_between_neurons = (max_x - min_x) / number_of_neurons_in_widest_layer
    for layer_num, neurons_list in neurons_mapping.items():
        min_x = horizontal_distance_between_neurons * (
                number_of_neurons_in_widest_layer - num_neurons_in_layers[layer_num]) / 2
        for community in neurons_list:
            for neuron in community:
                coordinates[neuron] = (min_x, min_y)
                min_x += horizontal_distance_between_neurons
        min_y += vertical_distance_between_layers
    return coordinates


def plot_no_comunites(
        organism: Organism,
        min_x: float = 30,
        max_x: float = 600,
        min_y: float = 20,
        max_y: float = 300,
):
    coordinates = {}
    num_neurons_in_layers = organism.dimensions
    number_of_neurons_in_widest_layer = num_neurons_in_layers[0]
    vertical_distance_between_layers = (max_y - min_y) / len(num_neurons_in_layers)
    horizontal_distance_between_neurons = (max_x - min_x) / number_of_neurons_in_widest_layer
    for layer_num in range(len(organism.dimensions)):
        min_x = horizontal_distance_between_neurons * (
                number_of_neurons_in_widest_layer - num_neurons_in_layers[layer_num]) / 2
        for neuron in organism.layer_neuron_idx_mapping[layer_num]:
            coordinates[neuron] = (min_x, min_y)
            min_x += horizontal_distance_between_neurons
        min_y += vertical_distance_between_layers
    widths = nx.get_edge_attributes(organism.network, 'weight')
    if not widths:
        line_width = [1 / 2 for _ in range(len(organism.network.edges))]
        edgelist = list(organism.network.edges)
    else:
        line_width = [x / 2 for x in list(widths.values())]
        edgelist = widths.keys()
    nx.draw(organism.network, coordinates, with_labels=True, arrows=False, edgelist=edgelist,
            width=line_width, node_color=COLORS[0])
    plt.show()
