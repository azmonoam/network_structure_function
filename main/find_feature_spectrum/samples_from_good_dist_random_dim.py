from datetime import datetime as dt
from pathlib import Path

import joblib
import networkx as nx

from find_arch_from_features_genrtic_random_dim import FindArchGeneticRandomDim
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from find_feature_spectrum.find_feature_dist_utils import get_selected_feature_names
from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim

if __name__ == '__main__':
    all_features = False
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = "/Volumes/noamaz/modularity"
    train_test_folder_name = 'train_test_data'
    task_name = 'xor'
    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    exp_name = selected_params.feature_selection_folder
    train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
    lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
    samples_path = f"{train_test_dir}/{selected_params.selected_features_data}"
    used_features_csv_path = f"{lgb_dir}/{exp_name}/{selected_params.used_features_csv}"
    folder_name = f'{num_features}_features'

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
        task_class = RetinaByDim
    elif task_name == 'xor':
        dims = [6, 6, 5, 3, 2]
        num_layers = len(dims) - 1
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
        task_class = XoraByDim
    else:
        raise ValueError
    potential_parents_percent = 15
    population_size = 1000
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

    find_arches_genetic = FindArchGeneticRandomDim(
        generations=generations,
        population_size=population_size,
        potential_parents_percent=potential_parents_percent,
        selected_feature_names=selected_feature_names,
        target_feature_values=target,
        use_distance_fitness=True,
        find_feature_dist=sampler,
        task_params_example=task_params,
        task_class=task_class,
    )
    orgs_to_save = find_arches_genetic._get_x_archs_with_features_genetic(
        num_orgs_to_return=2000,
        distance_early_stopping_criteria_num_sigmas=0.5,
        errors=dist_stds,
        test_saved_orgs_error=True,
    )
    valid_orgs_to_save = [o for o in orgs_to_save if len(list(nx.isolates(o.network))) == 0]
    graphs_and_dims = [
        [
            nx.to_numpy_array(g.network, weight=None, dtype=int),
            g.dimensions,
        ]
        for g in valid_orgs_to_save
    ]
    data_to_save = {
        'graphs': graphs_and_dims,
        'target_feature_values': target,
        'selected_feature_names': selected_feature_names,
        'errors': dist_stds,
    }

    num_features_record_folder = Path(f"{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/")
    num_features_record_folder.mkdir(exist_ok=True)
    record_folder = Path(f"{num_features_record_folder}/good_archs")
    record_folder.mkdir(exist_ok=True)
    models_folder = Path(f"{record_folder}/teach_archs_models")
    results_folder = Path(f"{record_folder}/teach_archs_results")
    models_folder.mkdir(exist_ok=True)
    results_folder.mkdir(exist_ok=True)
    with open(
            f"{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/{time_str}_data_for_{num_features}_features_good_archs.pkl",
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

    print('a')
