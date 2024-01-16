from _datetime import datetime as dt

import joblib

from ergm_model.data_halper import get_dist_lookup_data
from find_arch_from_features_genrtic import FindArchGenetic
from jobs_params import get_new_arch_params
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
    num_features = 10
    use_distance_fitness = False
    mse_early_stopping_criteria_factor = 0.005
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    #base_path = '/Volumes/noamaz/modularity'

    selected_feature_names, find_feature_dist = get_dist_lookup_data(
        task=task,
        num_features=num_features,
        base_path=base_path,
        normalize=True,
        params_dict=get_new_arch_params,
    )
    connectivity_ratio = round(
        find_feature_dist.target_mean_features[selected_feature_names.index('connectivity_ratio')], 2
    )
    print(f'target values: {find_feature_dist.target_mean_features}')
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
    )
    for ind, org in enumerate(orgs_to_save):
        with open(
                f'{base_path}/teach_archs/{task}/{task}_teach_archs_requiered_features_genetic/{num_features}_features/models/{exp_name}_{ind}.pkl',
                'wb+') as fp:
            joblib.dump(org, fp)
    print(f'saved {len(orgs_to_save)} orgs')
