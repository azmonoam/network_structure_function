import argparse
import os

import joblib
import numpy as np

from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_features', default=6)
    parser.add_argument('--point', default=2)
    parser.add_argument('-f', '--folder_name', default='6_features_nice')

    args = parser.parse_args()
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
    #base_path = '/Volumes/noamaz/modularity'
    folder_path = "retina_xor/retina_3_layers_3_4"
    points = [
        '2023-09-13-16-19-39_J12_P1_not_conv',
        '2023-09-13-17-29-22_J2_P1',
        '2023-09-13-17-29-22_J5_P1',
    ]

    point_name = points[point]
    point_path = f'{base_path}/{folder_path}/ergm_results/{folder_name}/{point_name}'
    point_samples_path = f'{point_path}/samples'
    all_mean_g_stasts = []

    for ergm_sample in os.listdir(point_samples_path):
        with open(f"{point_samples_path}/{ergm_sample}", 'rb') as fp:
            ergm_sample = joblib.load(fp)
        all_mean_g_stasts.append(ergm_sample['mean_graphs_stats'])

    mean_mean_g_stats = np.mean(all_mean_g_stasts, axis=0)
    std_mean_g_stats = np.std(all_mean_g_stasts, axis=0)
    res = {
        'mean_mean_g_stats': mean_mean_g_stats,
        'std_mean_g_stats': std_mean_g_stats,
        'all_mean_g_stasts': all_mean_g_stasts,
    }
    with open(
            f'{point_path}/{point_name}_agg_resample.pkl',
            'wb+') as fp:
        joblib.dump(res, fp)
