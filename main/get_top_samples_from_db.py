import random
from datetime import datetime as dt

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from ergm_model.data_halper import get_list_of_functions_from_features_names
from ergm_model.nx_methods import NxMethods
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from find_feature_spectrum.find_feature_dist_utils import get_selected_feature_names
from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim
from parameters.digits.digits_by_dim import DigitsByDim
from utils.main_utils import get_all_possible_dims

if __name__ == '__main__':
    all_features = False
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = "/Volumes/noamaz/modularity"
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    train_test_folder_name = 'train_test_data'
    task_name = 'digits'
    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    exp_name = selected_params.feature_selection_folder
    first_analsis_csv_name = selected_params.first_analsis_csv
    train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
    lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
    org_folder = f"{base_path}/{folder_path}/teach_archs_models"
    samples_path = f"{train_test_dir}/{selected_params.selected_features_data}"
    used_features_csv_path = f"{lgb_dir}/{exp_name}/{selected_params.used_features_csv}"
    const_feature = [
        'dimensions_0',
        'dimensions_1',
        'dimensions_2',
        'dimensions_3',
        'max_possible_connections',
        'num_layers',
        'num_neurons'
    ]
    features_to_multipl = [
        'total_connectivity_ratio_between_layers_0',
        'total_connectivity_ratio_between_layers_1',
        'total_connectivity_ratio_between_layers_2',
        'total_connectivity_ratio_between_layers_3',
        'max_connectivity_between_layers_per_layer_0',
        'max_connectivity_between_layers_per_layer_1',
        'max_connectivity_between_layers_per_layer_2',
        'max_connectivity_between_layers_per_layer_3',
        'connectivity_ratio'
    ]
    possible_number_of_neurons_in_layer = None
    if task_name == 'retina_xor':
        dims = [6, 3, 4, 2]
        num_layers = len(dims) - 1
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
    elif task_name == 'xor':
        dims = [6, 6, 5, 3, 2]
        num_layers = len(dims) - 1
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
    elif task_name == 'digits':
        col = 'Greens'
        dims = [64, 6, 6, 10]
        num_layers = len(dims) - 1
        task_params = DigitsByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
        possible_number_of_neurons_in_layer = range(2, 11)
    else:
        raise ValueError
    frec_of_good = 0.1
    sampler = FindFeaturesDistByPerformance(
        num_features=num_features,
        samples_path=samples_path,
        min_range_ind=-2,
        max_range_ind=-1,
    )
    selected_feature_names = get_selected_feature_names(
        used_features_csv_name=used_features_csv_path,
        num_features=num_features
    ).to_list()
    dist_stds = sampler.get_errors(
        num_features=num_features,
        frec=frec_of_good,
    )
    target = sampler.target_mean_features
    methods = NxMethods()
    features = get_list_of_functions_from_features_names(
        method_class=methods,
        features_names_list=selected_feature_names,
    )
    print(f"target_value: {target}")
    print(f"dist_stds: {dist_stds}")
    first_analsis = pd.read_csv(f"{base_path}/{folder_path}/first_analysis_results/{first_analsis_csv_name}")
    exps = first_analsis[
        first_analsis['mean_performance'].between(sampler.labels_range[0] / 1000, sampler.labels_range[1] / 1000)][
        'exp_name']
    graphs_and_dims = []
    all_features_values = []
    graphs_per_dim = {
        dim: {
            "all_features_values": [],
            "graphs": [],
        }
        for dim in
        get_all_possible_dims(task_params.input_dim, task_params.output_dim, task_params.num_layers,
                              task_params.num_neurons, possible_number_of_neurons_in_layer)
    }
    features_inds_to_remove = [
        selected_feature_names.index(f)
        for f in const_feature
        if f in selected_feature_names
    ]
    features_inds_to_mult = [
        selected_feature_names.index(f)
        for f in features_to_multipl
        if f in selected_feature_names
    ]
    for exp_name in exps:
        with open(f"{org_folder}/{exp_name}.pkl", 'rb+') as fp:
            org = joblib.load(fp)
        dim = org.dimensions
        graph = nx.to_numpy_array(org.network, weight=None, dtype=int)
        features_values = [
            f(org, f_name)
            for f, f_name in zip(features, selected_feature_names)
        ]
        all_features_values.append(features_values)
        graphs_and_dims.append(
            [
                graph,
                dim,
                features_values,
            ]
        )
        graphs_per_dim[tuple(dim)]["graphs"].append(graph)
        for ind in features_inds_to_mult:
            features_values[ind] = features_values[ind] * 100
        for ind in features_inds_to_remove:
            features_values.pop(ind)
        graphs_per_dim[tuple(dim)]["all_features_values"].append(features_values)

    data_to_save = {
        'graphs': graphs_and_dims,
        'target_feature_values': target,
        'selected_feature_names': selected_feature_names,
        'errors': dist_stds,
    }

    with open(
            f"{base_path}/{folder_path}/ergm/{num_features}_features/{time_str}_all_good_archs_from_db_data_and_graph.pkl",
            'wb+') as fp:
        joblib.dump(data_to_save, fp)

    for dim_data in graphs_per_dim.values():
        num_graphs = len(dim_data['all_features_values'])
        dim_data['num_graphs'] = num_graphs
        if num_graphs == 0:
            continue
        dim_data['target_feature_values'] = np.mean(dim_data['all_features_values'], axis=0)
        means = np.zeros((100, len(features_values)))
        for i in range(100):
            means[i] = np.mean(
                random.sample(
                    dim_data['all_features_values'],
                    int(num_graphs * frec_of_good),
                )
                , axis=0)
        dim_data['errors'] = np.std(means, axis=0)
    data_to_save["data_per_dim"] = graphs_per_dim
    with open(
            f"{base_path}/{folder_path}/ergm/{num_features}_features/{time_str}_all_good_archs_from_db_data_and_graph_per_dim.pkl",
            'wb+') as fp:
        joblib.dump(data_to_save, fp)

    num_samples_to_sample = int(len(graphs_and_dims) * frec_of_good * 1.5)
    stop = False
    i = 0
    while not stop and i < num_samples_to_sample:
        chosen = random.sample(graphs_and_dims, num_samples_to_sample)
        f_ = [f for _, _, f in chosen]
        chosen_g_errors = [
            abs(mean_obs_stat - target) / std
            for mean_obs_stat, target, std
            in zip(np.mean(f_, axis=0), target, dist_stds)
        ]
        if sum(1 for error in chosen_g_errors if error < 1) == len(features):
            stop = True
        i += 1
    data_to_save = {
        'graphs': chosen,
        'target_feature_values': target,
        'selected_feature_names': selected_feature_names,
        'errors': dist_stds,
    }
    with open(
            f"{base_path}/{folder_path}/ergm/{num_features}_features/{time_str}_chosen_{num_samples_to_sample}_good_archs_from_db_data_and_graph.pkl",
            'wb+') as fp:
        joblib.dump(data_to_save, fp)
