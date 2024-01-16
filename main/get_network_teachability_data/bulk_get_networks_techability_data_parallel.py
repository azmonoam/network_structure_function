import argparse
import os
from datetime import datetime
from typing import Tuple, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load

from parameters.retina_parameters import retina_structural_features_full_name_vec

WANTED_CONNECTIVITY_KEYS = [
    'total_connectivity_ratio_between_layers',
    'max_connectivity_between_layers_per_layer',
    'layer_connectivity_rank',
    'num_paths_to_output_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
]


def _check_if_name_in_wanted_name_catagories(
        name: str,
) -> bool:
    if name.rsplit('_', 1)[0] in WANTED_CONNECTIVITY_KEYS:
        return True
    return False


def _build_tabels_rows(
        csv_name: str,
        extended_results: bool,
        add_motifs: bool,
        new_models: bool,
) -> Optional[Dict[str, Any]]:
    print(datetime.now())
    threshold = 1.0
    try:
        results = pd.read_csv(f"{folder}/{csv_name}").drop("Unnamed: 0", axis=1, errors='ignore')
    except:
        print(f"{folder}/{csv_name}")
        return
    if results.shape[0] <= 1:
        return
    exp_name = results['exp_name'].iloc[0]
    with open(f"{models_folder}/{exp_name}.pkl", 'rb') as fp:
        organism = load(fp)

    modularity = organism.structural_features.modularity.modularity
    num_connections = organism.structural_features.connectivity.num_connections
    connectivity_ratio = organism.structural_features.connectivity.connectivity_ratio
    entropy = organism.structural_features.entropy.entropy
    normed_entropy = organism.structural_features.entropy.normed_entropy
    final_epoch_res = results[results['iterations'] == results['iterations'].max()]
    if final_epoch_res.shape[0] > 0:
        num_successes = final_epoch_res[final_epoch_res['performance'] >= threshold].shape[0]
        first_analysis = {
            'exp_name': exp_name,
            'modularity': modularity,
            'num_connections': num_connections,
            'entropy': entropy,
            'normed_entropy': normed_entropy,
            'median_performance': final_epoch_res['performance'].median(),
            'mean_performance': final_epoch_res['performance'].mean(),
            'performance_std': final_epoch_res['performance'].std(),
            'max_performance': final_epoch_res['performance'].max(),
            'connectivity_ratio': connectivity_ratio,
            f'num_successes_1.0': num_successes,
            f'success_percent_1.0': num_successes / final_epoch_res.shape[0]
        }
        if new_models:
            for i, motif in enumerate(organism.structural_features.motifs.motifs_count):
                first_analysis[f'motifs_count_{i}'] = motif
            for i, motif in enumerate(organism.normed_structural_features.motifs.motifs_count):
                first_analysis[f'normalized_motifs_count_{i}'] = motif
            for i, dim in enumerate(organism.structural_features.structure.dimensions):
                first_analysis[f'neurons_in_layer_{i}'] = dim
            first_analysis['num_layers'] = organism.structural_features.structure.num_layers
            first_analysis['num_neurons'] = organism.structural_features.structure.num_neurons
            first_analysis[
                'max_possible_connections'] = organism.structural_features.connectivity.max_possible_connections
            first_analysis['normalized_entropy'] = organism.normed_structural_features.entropy.entropy
            first_analysis['normalized_normed_entropy'] = organism.normed_structural_features.entropy.normed_entropy
        elif add_motifs:
            for i, motif in enumerate(organism.structural_features.motifs.motifs_count):
                first_analysis[f'motifs_count_r_{i}'] = motif
        if extended_results:
            connectivity_dict = _get_extended_results(organism)
            return {**first_analysis, **connectivity_dict}
        print(datetime.now())
        return first_analysis
    return None


def _get_extended_results(organism):
    connectivity_dict = {
        full_name_vec[ind]: value
        for ind, value in enumerate(organism.structural_features.connectivity.get_class_values())
        if _check_if_name_in_wanted_name_catagories(full_name_vec[ind])
    }
    connectivity_dict['mean_num_paths_to_output_per_input_neuron'] = \
        np.mean(organism.structural_features.connectivity.num_paths_to_output_per_input_neuron)
    connectivity_dict['mean_num_involved_neurons_in_paths_per_input_neuron'] = \
        np.mean(organism.structural_features.connectivity.num_involved_neurons_in_paths_per_input_neuron)
    connectivity_dict['num_neuron_combinations_that_have_a_connecting_path'] = sum(
        1
        for distance in organism.structural_features.connectivity.distances_between_input_neuron
        if distance != -1
    )
    connectivity_dict['num_neuron_combinations_that_dont_have_a_connecting_path'] = sum(
        1
        for distance in organism.structural_features.connectivity.distances_between_input_neuron
        if distance == -1
    )
    connectivity_dict['ratio_of_num_neuron_combinations_that_have_dont_have_a_connecting_path'] = (
        connectivity_dict['num_neuron_combinations_that_have_a_connecting_path'] /
        connectivity_dict['num_neuron_combinations_that_dont_have_a_connecting_path']
        if connectivity_dict['num_neuron_combinations_that_dont_have_a_connecting_path'] != 0 else 10
    )
    connectivity_dict['mean_distances_between_input_neuron'] = np.mean(
        [
            distance
            for distance in organism.structural_features.connectivity.distances_between_input_neuron
            if distance != -1
        ]
    )
    return connectivity_dict


def _prepare_data_from_csv_wrapper(
        num_cores: int,
        folder: str,
        extended_results: bool,
        add_motifs: bool,
        new_models: bool,
        job_num: int,
        bulk_size: int = 10000,
) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    print(f"-- using {folder} as folder name --")
    file_list = sorted(os.listdir(path=folder))
    inds = list(range(0, int(np.ceil(len(file_list) / bulk_size) * bulk_size) + bulk_size, bulk_size))
    i, j = [(inds[i], inds[i + 1]) for i in range(len(inds) - 1)][job_num - 1]
    return Parallel(
        n_jobs=num_cores,
        timeout=9999,
    )(
        delayed
        (_build_tabels_rows)(csv_name, extended_results, add_motifs, new_models)
        for csv_name in file_list[i:j]
        if os.path.exists(f"{folder}/{csv_name}")
    )


if __name__ == '__main__':
    extended_results = False
    add_motifs = False
    new_models = True
    base_path = '/home/labs/schneidmann/noamaz/modularity/'
    # base_path = '/Volumes/noamaz/modularity'
    full_name_vec = retina_structural_features_full_name_vec[3:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_folder', default='teach_archs_results')
    parser.add_argument('--models_folder', default='teach_archs_models')
    parser.add_argument('--base_folder', default='retina/dynamic_retina_3_layers')
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--job_num', default='1')

    args = parser.parse_args()

    job_num = int(args.job_num)
    base_folder = args.base_folder
    res_folder = args.res_folder
    models_folder = f"{base_path}/{base_folder}/{args.models_folder}"
    num_cores = int(args.n_threads)
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    folder = f"{base_path}/{base_folder}/{res_folder}"
    print(f"-- using {num_cores} cores --")

    columns = [
        'exp_name',
        'modularity',
        'entropy',
        'normed_entropy',
        'num_connections',
        'median_performance',
        'mean_performance',
        'max_performance',
        'performance_std',
        'connectivity_ratio',
        'num_successes_1.0',
        'success_percent_1.0',
        'motifs_count_0',
        'motifs_count_1',
        'motifs_count_2',
    ]
    first_analysis_df = pd.DataFrame(
        columns=columns,
    )
    data_list = _prepare_data_from_csv_wrapper(
        num_cores=num_cores,
        folder=folder,
        extended_results=extended_results,
        add_motifs=add_motifs,
        new_models=new_models,
        job_num=job_num,
    )
    for first_analysis_dict in data_list:
        if first_analysis_dict is not None:
            first_analysis_df = pd.concat([first_analysis_df, pd.DataFrame(first_analysis_dict, index=[0], )],
                                          ignore_index=True)
    if extended_results:
        csv_name = f'{time_str}_all_extended_results_from_{res_folder}'
    elif add_motifs:
        csv_name = f'{time_str}_all_results_from_{res_folder}_with_motifs'
    else:
        csv_name = f'{time_str}_all_results_from_{res_folder}'

    first_analysis_df.to_csv(f'{base_path}/{base_folder}/{csv_name}_{job_num}.csv')
