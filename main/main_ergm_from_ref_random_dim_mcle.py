import argparse
from _datetime import datetime as dt
from pathlib import Path

import joblib
from ergm_model.random_dim.ergm_random_dim_mcle import ErgmRandomdimMCLE
from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=4)
    parser.add_argument('--job_num', default=11)
    parser.add_argument('--n_iterations', default=20000)
    parser.add_argument('--n_std', default=1)
    parser.add_argument('--loose', default=1)
    parser.add_argument('--n_g', default=300)
    parser.add_argument('--task_name', default='xor')

    args = parser.parse_args()
    n_threads = int(args.n_threads)
    job_num = int(args.job_num)
    n_iterations = int(args.n_iterations)
    n_std = int(args.n_std)
    loose = bool(int(args.loose))
    num_graphs_to_sample_for_early_stopping = int(args.n_g)
    task_name = args.task_name

    alphas = {
        1: [0.0002, 0.0001],
        2: [0.00008, 0.00008],
        3: [0.0001, 0.00008],
        4: [0.0001, 0.00007],
        5: [0.001, 0.0001],
        6: [0.005, 0.0005],
        7: [0.008, 0.0005],
        8: [0.01, 0.0001],
        9: [0.0004, 0.0001],
        10: [0.0005, 0.0001],
        11: [0.01, 0.00001],
    }
    stop_evreys = {
        1: 5,
        2: 5,
        3: 5,
        4: 5,
        5: 5,
        6: 5,
        7: 5,
        8: 5,
        9: 5,
        10: 5,
        11: 5,
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
    features_names = prepered_data['selected_feature_names']
    target_mean_features = prepered_data['target_feature_values']
    example_graphs = prepered_data['graphs']
    errors = prepered_data['errors']
    print(f"stds: {errors}")
    initial_coeffs_values = None
    e = ErgmRandomdimMCLE(
        features_names=features_names,
        restrict_allowed_connections=True,
        target_mean_features=target_mean_features,
        example_graphs=example_graphs,
        n_threads=n_threads,
        errors=errors,
        num_layers=selected_params.num_layers,
        n_nodes=selected_params.num_neurons,
        input_size=selected_params.input_size,
        output_size=selected_params.output_size
    )
    all_coefs, mean_recorded_coeff, graphs, graphs_stats = e.find_coefs(
        early_stopping_type='std',
        num_stds=n_std,
        start_alpha=start_alpha,
        end_alpha=end_alpha,
        n_iterations=1200,
        stop_every=stop_every,
        test_for_loose=loose,
        num_tests_for_alphas=150,
        last_changed_itter=600,
        use_cov_noise=False,
        num_graphs_to_sample_for_early_stopping=num_graphs_to_sample_for_early_stopping,
    )
    print(f'final mean_recorded_coeff: {mean_recorded_coeff}')
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
        'alphas': (start_alpha, end_alpha)
    }

    num_features_record_folder = Path(f'{base_path}/{folder_path}/ergm_results/{folder_name}/')
    num_features_record_folder.mkdir(exist_ok=True)

    with open(
            f'{num_features_record_folder}/{exp_name}_J{job_num}_good_archs.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
