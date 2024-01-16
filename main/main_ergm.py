import argparse
from _datetime import datetime as dt

import joblib

from ergm_model.data_halper import get_dist_lookup_data
from ergm_model.ergm_classic import ErgmClassic
from jobs_params import get_new_arch_params_no_modularity
from utils.tasks_params import RetinaParameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=1)
    parser.add_argument('--job_num', default=1)
    args = parser.parse_args()
    n_threads = int(args.n_threads)
    job_num = int(args.job_num)
    start_alphas = {
        1: 0.0002,
        2: 0.0003,
        3: 0.0004,
        4: 0.0005,
        5: 0.0001,
        6: 0.00025,
    }
    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    task_params = RetinaParameters
    task = task_params.task_name
    num_features = 5
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'
    example_graphs_path = f'{base_path}/teach_archs/retina/retina_teach_archs_requiered_features_genetic/5_features_no_modularity/2023-07-17-14-24-59_models.pkl'
    selected_feature_names, find_feature_dist = get_dist_lookup_data(
        task=task,
        num_features=num_features,
        base_path=base_path,
        params_dict=get_new_arch_params_no_modularity,
        normalize=True,
    )
    initial_coeffs_values = None
    e = ErgmClassic(
        features_names=selected_feature_names,
        dimensions=task_params.start_dimensions,
        restrict_allowed_connections=True,
        find_feature_dist=find_feature_dist,
        initial_coeffs_values=initial_coeffs_values,
        example_graphs_path=example_graphs_path,
        n_threads=n_threads,
    )
    all_coefs, mean_recorded_coeff, graphs, graphs_stats = e.find_coefs(
        stop_every=10,
        start_alpha=start_alphas[job_num],
        early_stopping_type='error',
    )
    print(f'final mean_recorded_coeff: {mean_recorded_coeff}')
    res = {
        'target_feature_values': e.target_mean_features,
        'features_names': e.features_names,
        'all_coefs': all_coefs,
        'mean_recorded_coeff': mean_recorded_coeff,
        'graphs': graphs,
        'graphs_stats': graphs_stats,
    }
    with open(
            f'{base_path}/teach_archs/{task}/{task}_teach_archs_requiered_features_ergm/{num_features}_features/ergm_results/{exp_name}.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
