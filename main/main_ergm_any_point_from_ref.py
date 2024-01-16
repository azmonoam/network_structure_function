import argparse
from _datetime import datetime as dt

import joblib
import numpy as np

from ergm_model.ergm_to_any_point_from_ref import ErgmAnyPointFromRef
from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=1)
    parser.add_argument('--job_num', default=12)
    parser.add_argument('--num_features', default=6)
    parser.add_argument('--point', default=1)
    parser.add_argument('--n_iterations', default=10000)
    parser.add_argument('--n_std', default=1)
    parser.add_argument('-f', '--folder_name', default='6_features_nice')

    args = parser.parse_args()
    n_threads = int(args.n_threads)
    job_num = int(args.job_num)
    num_features = int(args.num_features)
    point = int(args.point)
    n_iterations = int(args.n_iterations)
    n_std = int(args.n_std)
    folder_name = args.folder_name

    alphas = {
        1: [0.0002, 0.0001],
        2: [0.0003, 0.0001],
        3: [0.0004, 0.0001],
        4: [0.0005, 0.0001],
        5: [0.0001, 0.0001],
        6: [0.005, 0.0005],
        7: [0.008, 0.0005],
        8: [0.0001, 0.00005],
        9: [0.0001, 0.00008],
        10: [0.0001, 0.00007],
        12: [0.00005, 0.0001],
    }
    start_alpha, end_alpha = alphas[job_num]
    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    dims = [6, 3, 4, 2]
    num_layers = len(dims) - 1
    task_params = RetinaByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
        task_base_folder_name='retina_xor',
        rule=LogicalGates.XOR,
    )

    base_path = '/home/labs/schneidmann/noamaz/modularity'
    base_path = '/Volumes/noamaz/modularity'
    folder_path = "retina_xor/retina_3_layers_3_4"
    if folder_name == '6_features_nice':
        six_features_list = [
            f"{folder_name}/2023-09-12-21-01-58_data_for_6_features_means.pkl",
            f"{folder_name}/2023-09-13-15-35-03_data_for_6_features_0.pkl",
            f"{folder_name}/2023-09-13-15-35-03_data_for_6_features_1.pkl",
        ]
    else:
        six_features_list = [
            f"{folder_name}/2023-09-11-10-32-56_data_for_6_features_0.pkl",
            f"{folder_name}/2023-09-11-10-32-56_data_for_6_features_1.pkl",
            f"{folder_name}/2023-09-11-10-32-56_data_for_6_features_2.pkl",
            f"{folder_name}/2023-09-11-10-32-56_data_for_6_features_3.pkl",
            f"{folder_name}/2023-09-11-18-06-51_data_for_6_features_4.pkl",
        ]

    prepered_data_paths = {
        10: ['2023-09-10-14-26-17_data_for_10_features_0.pkl'],
        6: six_features_list
    }
    data_pkl = prepered_data_paths[num_features][point]
    prepered_data_path = f'{base_path}/{folder_path}/requiered_features_genetic_models/{data_pkl}'
    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    features_names = prepered_data['selected_feature_names']
    target_mean_features = prepered_data['target_feature_values']
    example_graphs = prepered_data['graphs']
    errors = prepered_data['errors']
    initial_coeffs_values = None
    e = ErgmAnyPointFromRef(
        features_names=features_names,
        dimensions=task_params.start_dimensions,
        restrict_allowed_connections=True,
        initial_coeffs_values=initial_coeffs_values,
        target_mean_features=target_mean_features,
        example_graphs=example_graphs,
        n_threads=4,
        errors=errors,
    )
    all_coefs, mean_recorded_coeff, graphs, graphs_stats = e.find_coefs(
        stop_every=4,
        start_alpha=start_alpha,
        end_alpha=end_alpha,
        early_stopping_type='std',
        num_stds=n_std,
        n_iterations=1000,
        num_graphs_to_sample_for_early_stopping=500,
        test_for_loose=False,
        burning=20,
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
    }
    print(f"mean_features: {np.mean(graphs_stats, axis=0)}")
    print(f"errors: {e.all_errors[-1]}")
    print(f"sum errors: {sum(e.all_errors[-1])}")
    with open(
            f'{base_path}/{folder_path}/ergm_results/{folder_name}/{exp_name}_J{job_num}_P{point}.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
