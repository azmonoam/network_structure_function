import argparse
import random
from _datetime import datetime as dt
from pathlib import Path

import joblib

from ergm_model.ergm_from_reference_graph_multi_g import ErgmReferenceGraphMultiObs
from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=4)
    parser.add_argument('--job_num', default=4)
    parser.add_argument('--n_iterations', default=1200)
    parser.add_argument('--n_std', default=1)
    parser.add_argument('--loose', default=1)
    parser.add_argument('--n_g', default=20)
    parser.add_argument('--task_name', default='digits')
    parser.add_argument('--dim_ind', default=3)
    parser.add_argument('--num_example_graphs', default=15)
    parser.add_argument('--max_g', default=1000)

    args = parser.parse_args()
    n_threads = int(args.n_threads)
    job_num = int(args.job_num)
    n_iterations = int(args.n_iterations)
    n_std = int(args.n_std)
    loose = bool(int(args.loose))
    num_graphs_to_sample_for_early_stopping = int(args.n_g)
    task_name = args.task_name
    dim_ind = int(args.dim_ind)
    num_example_graphs = int(args.num_example_graphs)
    max_g = int(args.max_g)

    all_dims = [
        (64, 8, 4, 10),
        (64, 5, 7, 10),
        (64, 6, 6, 10),
        (64, 7, 5, 10),
    ]
    alphas = {
        1: [0.0002, 0.0001],
        2: [0.00008, 0.00008],
        3: [0.0000008, 0.0000008],
        4: [0.000001, 0.000001],
        5: [0.001, 0.0001],
        6: [0.005, 0.0005],
        7: [0.008, 0.0005],
        8: [0.0000008, 0.0000008],
        9: [0.000001, 0.000001],
        10: [0.005, 0.00001],
        11: [0.01, 0.00001],
        12: [0.05, 0.00001],
        13: [0.05, 0.0001],
        14: [0.08, 0.00001],
    }
    stop_evreys = {
        1: 5,
        2: 5,
        3: 2,
        4: 2,
        5: 5,
        6: 5,
        7: 5,
        8: 2,
        9: 2,
        10: 5,
        11: 5,
        12: 5,
        13: 5,
        14: 5,
    }
    start_alpha, end_alpha = alphas[job_num]
    stop_every = stop_evreys[job_num]

    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    num_layers = selected_params.num_layers
    folder_name = f'{num_features}_features'
    data_pkl = selected_params.ergm_init_graphs_pkl_file_name
    prepered_data_path = f'{base_path}/{folder_path}/ergm/{folder_name}/{data_pkl}'
    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    dim = all_dims[dim_ind]
    features_names = prepered_data['selected_feature_names']
    i = features_names.index('max_possible_connections')
    features_names.pop(i)
    target_mean_features = prepered_data["data_per_dim"][dim]['target_feature_values'].tolist()
    example_graphs = prepered_data["data_per_dim"][dim]['graphs']
    errors = prepered_data["data_per_dim"][dim]['errors']
    print(f"stds: {errors}")
    initial_coeffs_values = None
    if len(example_graphs) > max_g:
        example_graphs = random.sample(example_graphs, max_g)
    e = ErgmReferenceGraphMultiObs(
        features_names=features_names,
        restrict_allowed_connections=True,
        target_mean_features=target_mean_features,
        example_graphs=example_graphs,
        n_threads=n_threads,
        errors=errors,
        n_nodes=selected_params.num_neurons,
        dimensions=dim,
        num_graphs_to_sample_for_early_stopping=num_graphs_to_sample_for_early_stopping,
        early_stopping_type='std',
        num_stds=n_std,
        num_example_graphs=num_example_graphs,
    )
    all_coefs, mean_recorded_coeff, graphs, graphs_stats, did_converge = e.find_coefs(
        start_alpha=start_alpha,
        end_alpha=end_alpha,
        n_iterations=n_iterations,
        stop_every=stop_every,
        test_for_loose=loose,
        num_tests_for_alphas=150,
        last_changed_itter=600,
        use_cov_noise=False,
    )
    print(f"converged: {did_converge}")
    print(f'final mean_recorded_coeff: {mean_recorded_coeff}')
    if did_converge:
        try:
            requiered_g = prepered_data["data_per_dim"][dim]['num_graphs']
            if len(graphs) < requiered_g:
                num_graphs_to_sample = int((requiered_g - len(graphs)) / num_example_graphs)
                graphs2, g_stats2 = e.sample_multiple(
                    coefs=mean_recorded_coeff,
                    num_graphs_to_sample=num_graphs_to_sample,
                )
                graphs += graphs2
                graphs_stats += g_stats2
        except Exception as e:
            print(f'couldnt sample more graphs {e}')
    res = {
        'target_feature_values': e.target_mean_features,
        'features_names': e.features_names,
        'all_coefs': all_coefs,
        'mean_recorded_coeff': mean_recorded_coeff,
        'graphs': graphs,
        'graphs_stats': graphs_stats,
        'all_erors': e.all_errors,
        'all_mean_obs_stats': e.all_mean_obs_stats,
        'all_mean_recorded_coeffs': e.all_mean_recorded_coeffs,
        'allowed_error': errors,
        'model': e,
        'alphas': (start_alpha, end_alpha),
        'dimensions': dim,
        'example_graphs': e.example_graphs,
    }

    num_features_record_folder = Path(f'{base_path}/{folder_path}/ergm/{folder_name}/per_dim_results')
    num_features_record_folder.mkdir(exist_ok=True)
    dim_st = ''
    for d in dim:
        dim_st += f"_{d}"
    dim_record_folder = Path(f'{num_features_record_folder}/{dim_st[1:]}')
    dim_record_folder.mkdir(exist_ok=True)
    if not did_converge:
        dim_record_folder = Path(f'{dim_record_folder}/no_conv')
        dim_record_folder.mkdir(exist_ok=True)
    with open(
            f'{dim_record_folder}/{exp_name}_J{job_num}_good_archs_dim{dim_st}.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
