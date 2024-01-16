from _datetime import datetime as dt

import joblib
import pandas as pd

from ergm_model.data_halper import get_dist_lookup_data
from find_arch_from_features_genrtic import FindArchGenetic
from jobs_params import get_new_arch_params_bottom_performance
from utils.tasks_params import RetinaParameters

if __name__ == '__main__':
    save_togther = False
    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    task_params = RetinaParameters
    task = task_params.task_name
    mutation_probabilities = {
        'connection_switch_mutation_probability': 0.2,
    }
    potential_parents_percent = 15
    population_size = 500
    generations = 1000
    num_features = 5
    use_distance_fitness = False
    mse_early_stopping_criteria_factor = 0.005
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'
    models_folder = f'{num_features}_features_bottom_performance'
    selected_feature_names, find_feature_dist = get_dist_lookup_data(
        params_dict=get_new_arch_params_bottom_performance,
        task=task,
        num_features=num_features,
        base_path=base_path,
        normalize=True,
    )
    connectivity_ratio = round(
        find_feature_dist.target_mean_features[selected_feature_names.index('connectivity_ratio')], 2
    )
    print(f'target labels range: {find_feature_dist.labels_range}')
    print(f'target values: {find_feature_dist.target_mean_features}')
    all_res_no_duplicates = pd.read_csv(
        f'{base_path}/teach_archs/{task}/retina_teach_archs_requiered_features_genetic/5_features_bottom_performance/2023-07-26-13-02-48_all_results_combined_no_duplicates.csv')
    all_res_no_duplicates = all_res_no_duplicates[[
        'modularity',
        'num_connections',
        'entropy',
        'normed_entropy',
        'density',
    ]].astype(float).to_records().tolist()
    duplicates_orgs_features_values = {t[1:] for t in all_res_no_duplicates}
    find_arches_genetic = FindArchGenetic(
        generations=generations,
        task_params=task_params,
        connectivity_ratio=connectivity_ratio,
        population_size=population_size,
        mutation_probabilities=mutation_probabilities,
        potential_parents_percent=potential_parents_percent,
        selected_feature_names=selected_feature_names,
        find_feature_dist=find_feature_dist,
        use_distance_fitness=False,
    )
    orgs_to_save = find_arches_genetic._get_x_archs_with_features_genetic(
        num_orgs_to_return=500,
        mse_early_stopping_criteria_factor=0.005,
        duplicates_orgs_features_values=duplicates_orgs_features_values,
    )
    for ind, org in enumerate(orgs_to_save):
        with open(
                f'{base_path}/teach_archs/{task}/{task}_teach_archs_requiered_features_genetic/{models_folder}/models/{exp_name}_{ind}.pkl',
                'wb+') as fp:
            joblib.dump(org, fp)
    print(f'saved {len(orgs_to_save)} orgs')
