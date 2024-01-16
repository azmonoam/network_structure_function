import random
from datetime import datetime as dt

import joblib
import numpy as np

from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim
from utils.main_utils import get_all_possible_dims

if __name__ == '__main__':
    all_features = False
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = "/Volumes/noamaz/modularity"
    # base_path = '/home/labs/schneidmann/noamaz/modularity'
    train_test_folder_name = 'train_test_data'
    task_name = 'xor'
    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    exp_name = selected_params.feature_selection_folder
    first_analsis_csv_name = selected_params.first_analsis_csv
    train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
    lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
    org_folder = f"{base_path}/{folder_path}/teach_archs_models"
    samples_path = f"{train_test_dir}/{selected_params.selected_features_data}"
    used_features_csv_path = f"{lgb_dir}/{exp_name}/{selected_params.used_features_csv}"
    folder_name = f'{num_features}_features'
    data_pkl = selected_params.ergm_init_graphs_pkl_file_name
    prepered_data_path = f'{base_path}/{folder_path}/ergm/{folder_name}/{data_pkl}'
    if task_name == 'retina_xor':
        dims = [6, 3, 4, 2]
        num_layers = len(dims) - 1
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
        task_class = RetinaByDim
    elif task_name == 'xor':
        dims = [6, 6, 5, 3, 2]
        num_layers = len(dims) - 1
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
        task_class = XoraByDim
    else:
        raise ValueError
    frec_of_good = 0.1

    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    graphs_per_dim = {
        dim: {
            "all_features_values": [],
            "graphs": [],
        }
        for dim in
        get_all_possible_dims(task_params.start_dimensions[0], task_params.start_dimensions[-1], task_params.num_layers,
                              task_params.num_neurons)
    }
    for graph, dim, features_values, in prepered_data['graphs']:
        graphs_per_dim[tuple(dim)]["graphs"].append(graph)
        if task_name == "xor":
            features_values[1] = features_values[1] * 100
            features_values = features_values[1:-1]
        graphs_per_dim[tuple(dim)]["all_features_values"].append(features_values)
    for dim_data in graphs_per_dim.values():
        num_graphs = len(dim_data['all_features_values'])
        dim_data['num_graphs'] = num_graphs
        if num_graphs == 0:
            continue
        dim_data['target_feature_values'] = np.mean(dim_data['all_features_values'], axis=0)
        means = np.zeros((100, len(features_values)))
        for i in range(100):
            means[i] = np.mean(
                random.sample(
                    dim_data['all_features_values'],
                    int(num_graphs * frec_of_good),
                )
                , axis=0)
        dim_data['errors'] = np.std(means, axis=0)
    prepered_data["data_per_dim"] = graphs_per_dim
    with open(
            f"{base_path}/{folder_path}/ergm/{num_features}_features/{time_str}_all_good_archs_from_db_data_and_graph_per_dim.pkl",
            'wb+') as fp:
        joblib.dump(prepered_data, fp)
