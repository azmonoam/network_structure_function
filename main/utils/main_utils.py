import datetime
import os
from pathlib import Path
from typing import Union, List, Tuple, Set, Optional
import itertools
import joblib
import networkx as nx
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from neuroevolution import Neuroevolution
from organism import Organism

NodeType = Tuple[int, int]
Logical = Union[bool, int]
COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f', "#553939", "#9080ff", "#7d8f69"]

blues = ["#63bff0", "#a7d5ed", "#0d88e6", ]
rads = ["#ea5545", "#e1a692", "#b30000", ]
greens = ["#466964", "#48b5c4", "#3c4e4b"]
light_oranges = ["#edbf33", "#ede15b", "#ef9b20", ]
oranges = ["#e14d2a", "#ffb55a", "#fd841f"]
purples = ["#9080ff", "#beb9db", "#5e569b"]
pinks = ["#df979e", "#e4bcad", '#c80064']
grays = ["#b3bfd1", "#d7e1ee", "#54504c"]
limes = ["#d0ee11", "#d0f400", "#c9e52f", ]
bright_pinks = ["#de25da", "#ff80ff", "#eb44e8", ]
turquoise = ["#599e94", "#6cd4c5", "#466964", ]
browns = ["#553939", "#a77979", "#704f4f", ]
sage = ["#7d8f69", "#a9af7e", "#557153", ]
strong_blue = ["#5170fd", "#aee1fc", "#4636fc"]
maroon = ["#900c3f", "#c70039", '#900c3f']


def get_organism_from_pkl(
        path: str,
) -> Organism:
    with open(path, 'rb') as fp:
        return joblib.load(fp)


def save_to_pkl(
        best_folder: str,
        generation: Union[int, str],
        this_generation_best: Organism,
):
    with open(f'{best_folder}/best_{generation}.pkl', 'wb+') as fp:
        joblib.dump(this_generation_best, fp)


def print_log(
        neuroevolution: Neuroevolution,
        generation: int,
        this_generation_best: Organism,
):
    if neuroevolution.optimization_method == 'performance_and_connection_cost':
        print(f'{generation}: performance: {this_generation_best.performance:.3f}, '
              f'connection cost: {this_generation_best.connection_cost}, '
              f'modularity: {this_generation_best.modularity:.3f}')
    else:
        print(f'{generation}: performance: {this_generation_best.performance:.3f}, '
              f'modularity: {this_generation_best.modularity:.3f}')


def get_organism_list_from_pkl_folder(
        folder: str,
) -> List[Organism]:
    organisms = []
    get_generation = lambda x: int(x.split('best_')[-1].split('.pkl')[0])
    for path in tqdm(sorted(os.listdir(path=folder), key=get_generation)):
        organisms.append(
            get_organism_from_pkl(
                path=f'{folder}/{path}',
            )
        )
    return organisms


def get_modularity_list(
        organisms: List[Organism]
) -> List[float]:
    return [
        organism.modularity
        for organism in organisms
    ]


def get_connection_cost_list(
        organisms: List[Organism]
) -> List[int]:
    return [
        organism.connection_cost
        for organism in organisms
    ]


def get_performance_list(
        organisms: List[Organism]
) -> List[float]:
    return [
        organism.performance
        for organism in organisms
    ]


def export_results_to_csv(
        organisms: List[Organism],
        experiment_folder: str,
        performances: Optional[List[float]] = None,
        csv_name: Optional[str] = 'results',
):
    modularity = get_modularity_list(
        organisms=organisms,
    )
    connection_cost = get_connection_cost_list(
        organisms=organisms,
    )
    if not performances:
        performances = get_performance_list(
            organisms=organisms,
        )
    pd.DataFrame.from_dict(
        {
            'performances': performances,
            'modularity': modularity,
            'connection_cost': connection_cost,
        }
    ).to_csv(f'{experiment_folder}/{csv_name}.csv')


def set_up_experiment_folders(
        experiment_name: str,
        base_path: str = '',
) -> Tuple[str, str]:
    record_folder = Path(f"{base_path}experiment_data")
    record_folder.mkdir(exist_ok=True)
    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") \
        if experiment_name is None else experiment_name
    current_experiment_folder = record_folder / experiment_name
    print(current_experiment_folder)
    if current_experiment_folder.is_dir():
        raise Exception('Experiment already exists')
    current_experiment_folder.mkdir(exist_ok=True)
    best_folder = current_experiment_folder / 'best_network'
    best_folder.mkdir(exist_ok=True)

    return str(best_folder), str(current_experiment_folder)


def get_number_of_connections(
        organism: Organism,
) -> int:
    return organism.network.number_of_edges()


def check_path_to_output(
        output_nodes: List[NodeType],
        all_nodes_but_outputs: Set[NodeType],
        network: nx.DiGraph,
) -> List[NodeType]:
    nodes_with_path_to_output = []
    for node in all_nodes_but_outputs:
        if any(nx.has_path(network, node, x) for x in output_nodes):
            nodes_with_path_to_output.append(node)
    return nodes_with_path_to_output


def _remove_nodes_without_path_to_output(
        output_nodes: List[NodeType],
        network: nx.DiGraph,
) -> Set[NodeType]:
    all_nodes_but_outputs = set(network.nodes) - set(output_nodes)
    nodes_with_path_to_output = check_path_to_output(
        output_nodes=output_nodes,
        all_nodes_but_outputs=all_nodes_but_outputs,
        network=network,
    )
    nodes_to_remove = all_nodes_but_outputs - set(nodes_with_path_to_output)
    return nodes_to_remove


def clean_graph_from_unused_nodes(
        output_layer: int,
        input_layer: int,
        network: nx.DiGraph,
) -> nx.DiGraph:
    unused_nodes = get_all_unused_nodes(
        output_layer=output_layer,
        input_layer=input_layer,
        network=network
    )
    network.remove_nodes_from(
        unused_nodes
    )
    return network


def get_unconnected_nodes(
        network: nx.DiGraph,
        input_layer: int,
) -> Set[NodeType]:
    input_nodes = get_nodes_from_specific_layer(
        layer_ind=input_layer,
        network=network,
    )
    nodes_not_getting_anything = {
        node
        for node, degree in dict(network.in_degree()).items()
        if degree == 0 and node not in input_nodes
    }
    return set(nx.isolates(network)) | nodes_not_getting_anything


def get_nodes_from_specific_layer(
        layer_ind: int,
        network: nx.DiGraph,
) -> List[NodeType]:
    return [
        (layer, node_idx)
        for layer, node_idx in network.nodes
        if layer == layer_ind
    ]


def from_graph_layer_to_array_layer(
        layer: nx.DiGraph,
        layer_dim: Tuple[int, int]
) -> np.ndarray:
    layer = nx.to_numpy_array(layer)
    current_layer_num_neurons, next_layer_num_neurons = layer_dim
    return layer[:current_layer_num_neurons, -next_layer_num_neurons:]


def get_all_unused_nodes(
        output_layer: int,
        input_layer: int,
        network: nx.DiGraph,
) -> Set[NodeType]:
    output_nodes = get_nodes_from_specific_layer(
        layer_ind=output_layer,
        network=network,
    )
    nodes_without_path_to_output = _remove_nodes_without_path_to_output(
        output_nodes=output_nodes,
        network=network,
    )
    unconnected_nodes = get_unconnected_nodes(
        input_layer=input_layer,
        network=network
    )
    return nodes_without_path_to_output | unconnected_nodes


def compute_r2(
        target: torch.Tensor,
        output: torch.Tensor,
) -> torch.Tensor:
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2_score = 1 - ss_res / ss_tot
    return r2_score


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def softmax(
        x: List[float],
) -> List[float]:
    return np.exp(x) / np.sum(np.exp(x))


def get_all_possible_dims(
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_total_neurons: Optional[int] = None,
        possible_number_of_neurons_in_layer: Optional[int] = None,
):
    if possible_number_of_neurons_in_layer is None:
        possible_number_of_neurons_in_layer = range(min(input_size, output_size), max(input_size, output_size) + 1)
    number_of_missing_neurons = num_total_neurons - input_size - output_size
    num_layers_to_fill = num_layers - 1
    possible_mid_dims = {
        seq
        for seq in itertools.product(possible_number_of_neurons_in_layer, repeat=num_layers_to_fill)
        if sum(seq) == number_of_missing_neurons
    }
    possible_dims = [
        (
            input_size,
            *dim,
            output_size,
        )
        for dim in possible_mid_dims
    ]
    return possible_dims
