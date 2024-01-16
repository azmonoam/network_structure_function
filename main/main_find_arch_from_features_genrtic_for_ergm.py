from _datetime import datetime as dt

import joblib
import networkx as nx
from ergm_model.data_halper import get_dist_lookup_data
from find_arch_from_features_genrtic import FindArchGenetic
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
    #base_path = '/home/labs/schneidmann/noamaz/modularity'
    base_path = '/Volumes/noamaz/modularity'


    selected_feature_names, find_feature_dist = get_dist_lookup_data(
        task=task,
        num_features=num_features,
        base_path=base_path,
        normalize=True,
        avoid_modularity=True,
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
        num_orgs_to_return=10,
        mse_early_stopping_criteria_factor=0.005,
    )

    graphs = [
        nx.to_numpy_array(g.network, weight=None, dtype=int)
        for g in orgs_to_save
    ]
    with open(
            f'{base_path}/teach_archs/{task}/{task}_teach_archs_requiered_features_genetic/{num_features}_features_no_modularity/{exp_name}_models.pkl',
            'wb+') as fp:
        joblib.dump(graphs, fp)
    with open(
            f'{base_path}/teach_archs/{task}/{task}_teach_archs_requiered_features_genetic/{num_features}_features_no_modularity/{exp_name}_models_orgs.pkl',
            'wb+') as fp:
        joblib.dump(orgs_to_save, fp)
    print(f'saved {len(orgs_to_save)} orgs')
