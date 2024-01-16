import argparse
from _datetime import datetime as dt

import joblib

from ergm_model.ergm_classic import ErgmClassic
from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=1)
    parser.add_argument('--job_num', default=11)
    parser.add_argument('--num_features', default=6)
    parser.add_argument('--point', default=2)
    parser.add_argument('--n_iterations', default=5000)
    parser.add_argument('--n_std', default=1)
    parser.add_argument('--loose', default=1)
    parser.add_argument('-f', '--folder_name', default='6_features_nice')

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
        2: [0.0003, 0.0001],
        3: [0.0004, 0.0001],
        4: [0.0005, 0.0001],
        5: [0.0001, 0.0001],
        6: [0.005, 0.0005],
        7: [0.008, 0.0005],
        8: [0.0001, 0.00005],
        9: [0.0001, 0.00008],
        10: [0.0001, 0.00007],
        11: [0.00001, 0.00007],
    }
    stop_evreys = {
        1: 10,
        2: 10,
        3: 10,
        4: 10,
        5: 10,
        6: 5,
        7: 5,
        8: 10,
        9: 10,
        10: 10,
        11: 5,
    }
    start_alpha, end_alpha = alphas[job_num]
    stop_every = stop_evreys[job_num]
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
    #base_path = '/Volumes/noamaz/modularity'
    folder_path = "retina_xor/retina_3_layers_3_4"
    points = [
        '2023-09-13-16-19-39_J12_P1_not_conv',
        '2023-09-13-17-29-22_J2_P1',
        '2023-09-13-17-29-22_J5_P1',
    ]

    point_name = points[point]
    point_path = f'{base_path}/{folder_path}/ergm_results/{folder_name}/{point_name}'
    with open(f"{point_path}/{point_name}_agg_resample.pkl", 'rb') as fp:
        agg_data = joblib.load(fp)
    point_model_path = f'{base_path}/{folder_path}/ergm_results/{folder_name}/{point_name}.pkl'
    with open(point_model_path, 'rb') as fp:
        ergm_model = joblib.load(fp)
    features_names = ergm_model['features_names']
    target_mean_features = agg_data['mean_mean_g_stats'].tolist()
    example_graphs = ergm_model['graphs']
    errors =  agg_data['std_mean_g_stats']
    print(f"original coeffs: {ergm_model['mean_recorded_coeff']}")
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
        use_cov_noise=False,
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
            f'{base_path}/{folder_path}/ergm_results/{folder_name}/{point_name}/refind_ergm_{exp_name}_J{job_num}.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
