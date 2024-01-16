from datetime import datetime as dt
from pathlib import Path

import joblib
import networkx as nx

from find_arch_from_features_genrtic import FindArchGenetic
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from find_feature_spectrum.find_feature_dist_utils import get_selected_feature_names
from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.xor.xor_by_dim import XoraByDim

if __name__ == '__main__':
    all_features = False
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    num_features = 4
    base_path = "/Volumes/noamaz/modularity"
    train_test_folder_name = 'train_test_data'
    folder_name = f'{num_features}_features'
    task = 'xor'
    if task == 'retina':
        folder_path = "retina_xor/retina_3_layers_3_4"
        train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
        lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
        if num_features == 10:
            exp_name = 'exp_2023-09-09-13-55-25'
            connectivity_pkl_path = f"{train_test_dir}/retina_xor_2023-09-09-13-55-25_all_train_test_connectivity_data"
            samples_path = f"{train_test_dir}/retina_xor_2023-09-09-13-55-25_all_train_test_masked_data_10_features.pkl"
            used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-09-13-55-25_1_75_used_features.csv"
        elif num_features == 6:
            if all_features:
                exp_name = 'exp_2023-09-11-09-59-56'
                connectivity_pkl_path = f"{train_test_dir}/retina_xor_2023-09-11-09-59-56_all_train_test_connectivity_data"
                samples_path = f"{train_test_dir}/retina_xor_2023-09-11-09-59-56_all_train_test_masked_data_6_features.pkl"
                used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-11-09-59-56_1_70_used_features.csv"
                folder_name = '6_features_all_features'
            else:
                exp_name = 'exp_2023-09-12-20-38-43_nice_features'
                connectivity_pkl_path = f"{train_test_dir}/retina_xor_2023-09-12-20-38-43_all_train_test_connectivity_data_nice_features.pkl"
                samples_path = f"{train_test_dir}/retina_xor_2023-09-12-20-38-43_all_train_test_masked_data_6_features_nice_features.pkl"
                used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-12-20-38-43_1_60_used_features.csv"
                folder_name = '6_features_nice'
        elif num_features == 3:
            exp_name = 'exp_2023-09-14-16-31-14_nice_features'
            samples_path = f"{train_test_dir}/retina_xor_all_train_test_masked_data_3_top_features.pkl"
            used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-14-16-31-14_3_features_used_features.csv"
            folder_name = '3_top_features'
        dims = [6, 3, 4, 2]
        num_layers = len(dims) - 1
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
    elif task == 'xor':
        folder_path = "xor/xor_4_layers_6_5_3"
        train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
        exp_name = 'exp_2023-09-16-13-35-58_nice_features'
        if num_features == 4:
            lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
            samples_path = f"{train_test_dir}/2023-09-16-13-35-58_all_data_4_features_nica_features.pkl"
            used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-16-13-35-58_1_70_used_features.csv"
        dims = [6, 6, 5, 3, 2]
        num_layers = len(dims) - 1
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
    else:
        raise ValueError
    potential_parents_percent = 15
    population_size = 500
    generations = 1000
    use_distance_fitness = False
    mse_early_stopping_criteria_factor = 0.005

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
        frec=0.1,
    )
    target = sampler.target_mean_features
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
    )
    orgs_to_save = find_arches_genetic._get_x_archs_with_features_genetic(
        num_orgs_to_return=1000,
        distance_early_stopping_criteria_num_sigmas=1.0
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
    with open(
            f"{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/{time_str}_data_for_{num_features}_features_good_archs.pkl",
            'wb+') as fp:
        joblib.dump(data_to_save, fp)
    record_folder = Path(f"{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/good_archs")
    record_folder.mkdir(exist_ok=True)
    models_folder = Path(f"{record_folder}/teach_archs_models")
    results_folder = Path(f"{record_folder}/teach_archs_results")
    models_folder.mkdir(exist_ok=True)
    results_folder.mkdir(exist_ok=True)
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

    print('a')
