import argparse
from _datetime import datetime as dt

import joblib

from ergm_model.ergm_classic import ErgmClassic
from parameters.xor.xor_by_dim import XoraByDim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=4)
    parser.add_argument('--job_num', default=11)
    parser.add_argument('--num_features', default=4)
    parser.add_argument('--point', default=0)
    parser.add_argument('--n_iterations', default=20000)
    parser.add_argument('--n_std', default=1)
    parser.add_argument('--loose', default=1)
    parser.add_argument('-f', '--folder_name', default='4_features')

    args = parser.parse_args()
    n_threads = int(args.n_threads)
    job_num = int(args.job_num)
    num_features = int(args.num_features)
    point = int(args.point)
    n_iterations = int(args.n_iterations)
    n_std = int(args.n_std)
    loose = bool(int(args.loose))
    folder_name = args.folder_name

    alphas = {
        1: [0.0002, 0.0001],
        2: [0.00008, 0.00008],
        3: [0.0001, 0.00008],
        4: [0.0001, 0.00007],
        5: [0.0001, 0.0001],
        6: [0.005, 0.0005],
        7: [0.008, 0.0005],
        8: [0.0001, 0.00005],
        9: [0.0004, 0.0001],
        10: [0.0005, 0.0001],
        11: [0.001, 0.0001],
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
    dims = [6, 6, 5, 3, 2]
    num_layers = len(dims) - 1
    task_params = XoraByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )

    base_path = '/home/labs/schneidmann/noamaz/modularity'
    #base_path = '/Volumes/noamaz/modularity'
    folder_path = "xor/xor_4_layers_6_5_3"
    prepered_data_paths = {
        4: [
            '2023-09-17-14-03-07_data_for_4_features_0.pkl',
            '2023-09-18-12-09-38_data_for_4_features_good_archs.pkl'
        ],
    }
    data_pkl = prepered_data_paths[num_features][point]
    prepered_data_path = f'{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/{data_pkl}'
    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    features_names = prepered_data['selected_feature_names']
    target_mean_features = prepered_data['target_feature_values']
    example_graphs = prepered_data['graphs']
    errors = prepered_data['errors']
    print(f"stds: {errors}")
    initial_coeffs_values = None
    e = ErgmClassic(
        features_names=features_names,
        dimensions=task_params.start_dimensions,
        restrict_allowed_connections=True,
        initial_coeffs_values=initial_coeffs_values,
        target_mean_features=target_mean_features,
        example_graphs=example_graphs,
        n_threads=n_threads,
        errors=errors,
    )
    all_coefs, mean_recorded_coeff, graphs, graphs_stats = e.find_coefs(
        early_stopping_type='std',
        num_stds=n_std,
        start_alpha=start_alpha,
        end_alpha=end_alpha,
        n_iterations=n_iterations,
        stop_every=stop_every,
        test_for_loose=loose,
        num_tests_for_alphas=150,
        last_changed_itter=600,
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
    }
    with open(
            f'{base_path}/{folder_path}/ergm_results/{folder_name}/{exp_name}_J{job_num}_P{point}.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
