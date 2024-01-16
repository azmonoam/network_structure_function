import argparse

import joblib
import numpy as np

from ergm_model.ergm_classic import ErgmClassic
from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=1)
    parser.add_argument('--job_num', default=9)
    parser.add_argument('--num_features', default=6)
    parser.add_argument('--point', default=0)
    parser.add_argument('-f', '--folder_name', default='6_features_nice')

    args = parser.parse_args()
    n_threads = int(args.n_threads)
    job_num = int(args.job_num)
    num_features = int(args.num_features)
    point = int(args.point)
    folder_name = args.folder_name

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
    # base_path = '/Volumes/noamaz/modularity'
    folder_path = "retina_xor/retina_3_layers_3_4"
    points = [
        '2023-09-13-16-19-39_J12_P1_not_conv',
        '2023-09-13-17-29-22_J2_P1',
        '2023-09-13-17-29-22_J5_P1',
    ]

    point_name = points[point]
    point_model_path = f'{base_path}/{folder_path}/ergm_results/{folder_name}/{point_name}.pkl'
    with open(point_model_path, 'rb') as fp:
        ergm_model = joblib.load(fp)
    features_names = ergm_model['features_names']
    target_mean_features = np.mean(ergm_model['graphs_stats'], axis=0).tolist()
    coefs = ergm_model['mean_recorded_coeff']
    e = ErgmClassic(
        features_names=features_names,
        dimensions=task_params.start_dimensions,
        restrict_allowed_connections=True,
        target_mean_features=target_mean_features,
        n_threads=n_threads,
    )
    graphs, g_stats = e.sample_multiple(
        coefs=coefs,
        num_graphs_to_sample=20000,
    )
    res = {
        'target_feature_values': e.target_mean_features,
        'features_names': e.features_names,
        'graphs': graphs,
        'graphs_stats': g_stats,
        'mean_graphs_stats': np.mean(g_stats, axis=0),
    }
    print(f"mean_features: {np.mean(g_stats, axis=0)}")
    samples_folder = Path(f'{base_path}/{folder_path}/ergm_results/{folder_name}/{point_name}/samples')
    samples_folder.mkdir(exist_ok=True)
    with open(
            f'{samples_folder}/{point_name}_resample_20k_{job_num}.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
