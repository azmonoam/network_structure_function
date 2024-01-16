import argparse
from datetime import datetime as dt
from pathlib import Path

import joblib
import networkx as nx
import numpy as np

from find_arch_from_features_genrtic import FindArchGenetic
from find_feature_spectrum.find_feature_dist_by_performance_per_dim import FindFeaturesDistByPerformancePerDim
from logical_gates import LogicalGates
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--task_name', default='xor')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task_name = args.task_name

    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = "/Volumes/noamaz/modularity"
    train_test_folder_name = 'train_test_data'
    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    exp_name = selected_params.feature_selection_folder
    train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
    lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
    samples_path = f"{train_test_dir}/{selected_params.selected_features_data}"
    used_features_csv_path = f"{lgb_dir}/{exp_name}/{selected_params.used_features_csv}"
    folder_name = f'{num_features}_features'
    num_orgs_to_return = 1000
    if task_name == 'xor':
        all_dims = [
            (6, 5, 5, 4, 2),
            (6, 6, 4, 4, 2),
            (6, 5, 4, 5, 2),
            (6, 4, 6, 4, 2),
            (6, 6, 5, 3, 2),
            (6, 4, 5, 5, 2),
            (6, 5, 6, 3, 2),
            (6, 6, 3, 5, 2),
            (6, 4, 4, 6, 2),
            (6, 5, 3, 6, 2),
            (6, 3, 6, 5, 2),
            (6, 3, 5, 6, 2),
            (6, 6, 6, 2, 2),
            (6, 6, 2, 6, 2),
            (6, 2, 6, 6, 2),
        ]
        all_num_orgs_to_return = [
            2000,
            2000,
            1500,
            1500,
            1500,
            1500,
            1500,
            1000,
            1000,
            800,
            500,
            500,
            0,
            0,
            0,
        ]
        num_orgs_to_return = all_num_orgs_to_return[job_num - 1]
    elif task_name == 'retina_xor':
        all_dims = [
            (6, 5, 2, 2),
            (6, 2, 5, 2),
            (6, 3, 4, 2),
            (6, 4, 3, 2),
        ]
        all_num_orgs_to_return = [
            500,
            500,
            1500,
            2500,
        ]
        num_orgs_to_return = all_num_orgs_to_return[job_num - 1]
    elif task_name == 'digits':
        all_dims = [
            (64, 5, 7, 10),
            (64, 8, 4, 10),
            (64, 6, 6, 10),
            (64, 7, 5, 10),
        ]
        all_num_orgs_to_return = [
            1000,
            1000,
            4000,
            2000,
        ]
        num_orgs_to_return = all_num_orgs_to_return[job_num - 1]
    else:
        raise ValueError
    dim = all_dims[job_num - 1]
    data_pkl = selected_params.ergm_init_graphs_pkl_file_name
    prepered_data_path = f'{base_path}/{folder_path}/ergm/{folder_name}/{data_pkl}'
    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    target = prepered_data["data_per_dim"][dim]['target_feature_values'].tolist()
    example_graphs = prepered_data["data_per_dim"][dim]['graphs']
    dist_stds = prepered_data["data_per_dim"][dim]['errors']
    all_features_values = prepered_data["data_per_dim"][dim]['all_features_values']

    dim = list(dim)
    if task_name == 'retina_xor':
        num_layers = len(dim) - 1
        task_params = RetinaByDim(
            start_dimensions=dim,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
        task_class = RetinaByDim
        num_features = 5
        selected_feature_names = prepered_data['selected_feature_names']
        i = selected_feature_names.index('max_possible_connections')
        selected_feature_names.pop(i)
    elif task_name == 'xor':
        num_layers = len(dim) - 1
        task_params = XoraByDim(
            start_dimensions=dim,
            num_layers=num_layers,
            by_epochs=False,
        )
        task_class = XoraByDim
        num_features = 3
        selected_feature_names = prepered_data['selected_feature_names'][1:-1]
    elif task_name == 'digits':
        num_layers = len(dim) - 1
        task_params = DigitsByDim(
            start_dimensions=dim,
            num_layers=num_layers,
            by_epochs=False,
        )
        task_class = DigitsByDim
        num_features = 2
        selected_feature_names = prepered_data['selected_feature_names']
        i = selected_feature_names.index('max_possible_connections')
        selected_feature_names.pop(i)
    else:
        raise ValueError

    potential_parents_percent = 15
    population_size = 500
    generations = 1000
    use_distance_fitness = False
    mse_early_stopping_criteria_factor = 0.005
    sampler = FindFeaturesDistByPerformancePerDim(
        num_features=num_features,
        all_features_values=all_features_values,
    )
    print(f"target_value: {target}")
    print(f"dist_stds: {dist_stds}")

    find_arches_genetic = FindArchGenetic(
        generations=generations,
        task_params=task_params,
        population_size=population_size,
        potential_parents_percent=potential_parents_percent,
        selected_feature_names=selected_feature_names,
        target_feature_values=target,
        use_distance_fitness=True,
        find_feature_dist=sampler,
        connectivity_in_perc=True,
    )
    orgs_to_save = find_arches_genetic._get_x_archs_with_features_genetic(
        num_orgs_to_return=num_orgs_to_return,
        distance_early_stopping_criteria_num_sigmas=1.0,
        test_saved_orgs_error=False,
    )
    valid_orgs_to_save = [o for o in orgs_to_save if len(list(nx.isolates(o.network))) == 0]
    graphs = [
        nx.to_numpy_array(g.network, weight=None, dtype=int)
        for g in valid_orgs_to_save
    ]
    data_to_save = {
        'graphs': graphs,
        'target_feature_values': target,
        'selected_feature_names': selected_feature_names,
        'errors': dist_stds,
    }
    dim_st = ''
    for d in dim:
        dim_st += f"_{d}"
    dim_st = dim_st[1:]

    record_base_folder = Path(
        f"{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/good_archs/per_dim_results")
    record_base_folder.mkdir(exist_ok=True)
    record_folder = Path(f"{record_base_folder}/{dim_st}")
    record_folder.mkdir(exist_ok=True)
    models_folder = Path(f"{record_folder}/teach_archs_models_1s")
    results_folder = Path(f"{record_folder}/teach_archs_results_1s")
    models_folder.mkdir(exist_ok=True)
    results_folder.mkdir(exist_ok=True)

    with open(
            f"{record_folder}/{time_str}_data_for_{num_features}_features_good_archs.pkl",
            'wb+') as fp:
        joblib.dump(data_to_save, fp)

    fitness_values = []
    features_values = []
    feature_vals_by_feature_name = {
        k: []
        for k in selected_feature_names}
    for i, org in enumerate(valid_orgs_to_save):
        fitness_values.append(org.fitness.values[0])
        features_values.append(org.features_values)
        for j, f_val_name in enumerate(selected_feature_names):
            feature_vals_by_feature_name[f_val_name].append(org.features_values[j])
        with open(
                f"{models_folder}/{time_str}_{i}.pkl", 'wb+') as fp:
            joblib.dump(org, fp)
    print(f'saved {len(valid_orgs_to_save)} orgs')
    feature_values_dict = {
        'fitness_values': fitness_values,
        'features_values': features_values,
        'feature_vals_by_feature_name': feature_vals_by_feature_name,
    }
    with open(
            f"{record_folder}/{time_str}_feature_values_{num_features}_features_good_archs.pkl",
            'wb+') as fp:
        joblib.dump(feature_values_dict, fp)
    print(f"mean features values of saved orgs: {np.mean(features_values, axis=0)}")
    print(f"error of mean features values of saved orgs:")
    for i in range(num_features):
        print(abs(target[i] - np.mean(features_values, axis=0)[i]) / dist_stds[i])
    print('a')
