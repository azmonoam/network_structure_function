import os
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from archive.draw_nn import NeuralNetwork
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from plotting.plotting import plot_double_from_file, plot_performance_by_generation, _create_gif, draw_network_and_performance
from analsis.analsis_utils import rearrange_network_by_module, separate_to_sides, remove_unused_nodes_from_array


def from_graph_layer_to_array_layer(
        layer: nx.nx_agraph,
        layer_dim: Tuple[int, int]
) -> np.ndarray:
    layer = nx.to_numpy_array(layer)
    current_layer_num_neurons, next_layer_num_neurons = layer_dim
    return layer[:current_layer_num_neurons, -next_layer_num_neurons:]
exp_name = '2022-12-19-17-12-27'

folder = Path(f"/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/best_network")
path = 'best_509.pkl'
with open(f'{folder}/{path}', 'rb') as fp:
        organism = joblib.load(fp)
calc_structural_features = CalcStructuralFeatures(
            organism=organism,
        )
calc_structural_features.add_modularity()
exp_name = '2022-12-15_10_21_54'

best_organisms = []
performances = []
get_generation = lambda x: int(x.split('best_')[-1].split('.pkl')[0])
folder = Path(f"/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/best_network")
for path in tqdm(sorted(os.listdir(path=folder), key=get_generation)):
    with open(f'{folder}/{path}', 'rb') as fp:
        organism = joblib.load(fp)
    best_organisms.append(organism)
    performances.append(organism.performance)

for organism in best_organisms:
    layers = []
    for index, layer in enumerate(organism.layers):
        layer = from_graph_layer_to_array_layer(
            layer=layer,
            layer_dim=(organism.dimensions[index], organism.dimensions[index + 1])
        )
        layers.append(layer)
    organism.layers = layers
draw_network_and_performance(
    best_organisms=best_organisms,
    performances=performances,
    folder=f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}',
    optimization_method='performance_only',
    plot_every=15
)
plot_performance_by_generation(
    performances=performances,
    max_generations=len(performances),
    file_name=f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/performance',
    y_lim=[min(performances), max(performances) + 0.02],
    title=False,
)
plt.show()

exp_name = '2022-12-03_00:53:55'

with open(f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/all_with_mod.pkl',
          'rb') as fp:
    best_organisms = joblib.load(fp)

files_list = []
performances = [organism.performance for organism in best_organisms]
network_temp_file_name = f"/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/network_temp.png"
performance_temp_file_name = f"/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/performance_temp.png"
for i in range(len(best_organisms)):
    if i % 100 == 0:
        organism = best_organisms[i]
        performances.append(organism.performance)
        organism.layers = remove_unused_nodes_from_array(organism_layers=organism.layers)
        organism.layers = remove_unused_nodes_from_array(organism_layers=organism.layers)
        organism.layers = remove_unused_nodes_from_array(organism_layers=organism.layers)
        mapping_dict, new_left_community, new_right_community = separate_to_sides(
            organism=organism,
        )
        rearrange_network_by_module(
            organism=organism,
            mapping_dict=mapping_dict,
        )
        file_name = f'evolve_{i}_.png'
        files_list.append(file_name)
        network = NeuralNetwork(
            vertical_distance_between_layers=6,
            horizontal_distance_between_neurons=2,
            neuron_radius=0.5,
            number_of_neurons_in_widest_layer=8,
            communities=[new_left_community, new_right_community],
        )
        for layer in organism.layers:
            num_neurons = layer.shape[0]
            network.add_layer(num_neurons, layer.T)

        plot_performance_by_generation(
            performances=performances[:i],
            max_generations=len(performances),
            file_name=performance_temp_file_name,
            y_lim=[min(performances), max(performances) + 0.02],
            title=False,
        )
        network.draw(
            ind=i,
            fitness=organism.performance,
            file_name=network_temp_file_name,
            title=False,
        )
        plot_double_from_file(
            network_network_temp_file_namefile_name=network_temp_file_name,
            performance_temp_file_name=performance_temp_file_name,
            title=f'Generation: {i}, fitness: {organism.performance:.3f}',
            save_path=f"/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/{file_name}",
        )
sorted(files_list, key=lambda y: int(y.split('_')[1]))
_create_gif(
    sorted_files_list=files_list,
    folder=f"/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}",
)
