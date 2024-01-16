import argparse
from datetime import datetime as dt
from pathlib import Path
from stractural_features_models.structural_features import NICE_FEATURES, NICE_FEATURES_NO_INV

from jobs_params import regression_tree_num_features
from logical_gates import LogicalGates
from networks_teachability_regression.regression_tree_feature_selection_lightgbm import \
    LightGBMRegressionTreeFeatureSelection
from networks_teachability_regression.regression_tree_feature_selection_xboost import \
    XGBoostRegressionTreeFeatureSelection
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim
from utils.regression_methods import get_list_features_names

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='digits')
parser.add_argument('--regressor', default='lightgbm')
parser.add_argument('--n_threads', default=4)
parser.add_argument('--job_num', default=6)
parser.add_argument('--no_modularity', default=0)
parser.add_argument('--step', default=1)
parser.add_argument('--ind', default=1)

args = parser.parse_args()
task = args.task
n_threads = int(args.n_threads)
job_number = int(args.job_num)
regressor = args.regressor
no_modularity = bool(int(args.no_modularity))
step = int(args.step)
ind = int(args.ind)
print(f"ind: {ind}")
globality_list = [1, 0.75, 0.5, 0]
if task == 'digits':
    globality_ind = globality_list[ind]
else:
    globality_ind = globality_list[job_number - 1]
n_estimators = 500
features_to_drop = None
globality_level = {
    1: [
        'connectivity_ratio',
        'num_connections',
        'max_possible_connections',
        'motifs_count',
        'dimensions',
        'num_layers',
        'num_neurons'
    ],
    0.75: [
        'total_connectivity_ratio_between_layers',
        'max_connectivity_between_layers_per_layer',
        'layer_connectivity_rank',
    ],
    0.5: [
        'distances_between_input_neuron',
        'num_paths_to_output_per_input_neuron',
    ],
    0: [
        'out_connections_per_layer',
        'in_connections_per_layer',
    ]
}
if task != 'retina_xor':
    globality_level[0.5].append('num_involved_neurons_in_paths_per_input_neuron')

features_list = globality_level[globality_ind]

exp_folder_name_addition = f'_nice_features_glob_{globality_ind}'.replace(".", "_")
exp_folder = f'exp_2023-11-30-12-09-51_nice_features_glob_{globality_ind}'.replace(".", "_")
full_features_list = NICE_FEATURES
if task == 'xor':
    dims = [6, 6, 4, 4, 2]
    num_layers = len(dims) - 1
    task_params = XoraByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    random_archs = True
    if random_archs:
        task_params.task_version_name = f'{task}_{num_layers}_layers'
        name_ = 'random'
    else:
        name_ = tuple([d for d in dims[1:-1]])
    train_path = selected_exp_names[task][name_].train_data_path
    test_path = selected_exp_names[task][name_].test_data_path
    plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_base_folder_name}_multi_arch/{task_params.task_version_name}'
elif task == 'retina_xor':
    dims = [6, 5, 2, 2]
    num_layers = len(dims) - 1
    task_params = RetinaByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
        task_base_folder_name='retina_xor',
        rule=LogicalGates.XOR,
    )
    full_features_list = NICE_FEATURES_NO_INV
    random_archs = True
    if random_archs:
        task_params.task_version_name = f'retina_{num_layers}_layers'
        name_ = 'random'
    else:
        name_ = tuple([d for d in dims[1:-1]])
    train_path = selected_exp_names[task][name_].train_data_path
    test_path = selected_exp_names[task][name_].test_data_path
    plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_base_folder_name}_multi_arch/{task_params.task_version_name}'
elif task == 'digits':
    dims = [64, 6, 6, 10]
    num_layers = len(dims) - 1
    task_params = DigitsByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    random_archs = True
    if random_archs:
        task_params.task_version_name = f'digits_{num_layers}_layers'
        name_ = 'random'
    else:
        name_ = tuple([d for d in dims[1:-1]])
    train_path = selected_exp_names[task][name_].train_data_path
    test_path = selected_exp_names[task][name_].test_data_path
    exp_folder_name_addition = ''
    plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_base_folder_name}_multi_arch/{task_params.task_version_name}'
else:
    raise ValueError()

if __name__ == '__main__':
    plot_out = False
    plot_path = None
    task_data_folder_name = 'train_test_data'
    local_base_path = '/home/labs/schneidmann/noamaz/network_modularity'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = task_params.base_path
    # task_params.base_path = '/Volumes/noamaz/modularity'

    out_folder = f'{regressor}_feature_selection/by_globality'

    base_path_to_res = f"{task_params.teach_arch_base_path}/{task_data_folder_name}"
    out_dir = f"{task_params.teach_arch_base_path}/{out_folder}"
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    all_feature_names = get_list_features_names(
        task_params=task_params,
        features_list=full_features_list,
    )
    ind_to_drop = [i for i in range(len(all_feature_names))]
    for i, feature_name in enumerate(all_feature_names):
        for allowed_globality_feature_name in features_list:
            if allowed_globality_feature_name == 'connectivity_ratio' and feature_name != 'connectivity_ratio':
                continue
            if allowed_globality_feature_name in feature_name:
                ind_to_drop.remove(i)
                break

    if regressor == 'lightgbm':
        feature_selection_regression_tree = LightGBMRegressionTreeFeatureSelection(
            base_path_to_res=base_path_to_res,
            test_path=test_path,
            train_path=train_path,
            out_folder=out_folder,
            out_path=out_dir,
            time_str=time_str,
            task_params=task_params,
            n_threads=n_threads,
            n_estimators=100,
            features_to_drop=features_to_drop,
            features_list=features_list,
            exp_folder_name_addition=exp_folder_name_addition,
            exp_folder=exp_folder,
            ind_to_drop=ind_to_drop,
        )
    elif regressor == 'xgboost':
        record_folder = Path(f"{out_path}")
        feature_selection_regression_tree = XGBoostRegressionTreeFeatureSelection(
            base_path_to_res=base_path_to_res,
            test_path=test_path,
            train_path=train_path,
            out_folder=out_folder,
            out_path=out_dir,
            time_str=time_str,
            task_params=task_params,
            n_threads=n_threads,
            features_to_drop=features_to_drop,
            features_list=features_list,
            exp_folder_name_addition=exp_folder_name_addition,
            exp_folder=exp_folder,
            ind_to_drop=ind_to_drop,
        )
    else:
        raise ValueError
    feature_selection_regression_tree.eval_metric = 'mape'
    if task == 'digits':
        features_numbers = [job_number]
        calc_original = False
        res_df, models_df = feature_selection_regression_tree.regression_tree_feature_selection_non_parllel(
            features_numbers=features_numbers,
            step=step,
            calc_original=calc_original,
        )
    else:
        features_numbers = list(range(1, 10)) + list(
            range(10, len(feature_selection_regression_tree.feature_names), 10))
        calc_original = True
        res_df, models_df = feature_selection_regression_tree.regression_tree_feature_selection(
            features_numbers=features_numbers,
            step=step,
            calc_original=calc_original,
        )
    min_num_features = features_numbers[0]
    max_num_features = features_numbers[-1]
    res_df.to_csv(
        f"{feature_selection_regression_tree.exp_folder}/{time_str}_{min_num_features}_{max_num_features}_feature_selection.csv", )
    models_df.to_csv(
        f"{feature_selection_regression_tree.exp_folder}/{time_str}_{min_num_features}_{max_num_features}_used_features.csv", )
    if plot_out:
        if plot_path is None:
            plot_path = feature_selection_regression_tree._set_up_plot_folder(
                local_base_path=f'{local_base_path}/{out_folder}',
                allow_exp_folder_exist=True,
            )
        for metric in ['r2', 'mae', 'mape', 'mse']:
            feature_selection_regression_tree.plot_r2_vs_num_features(
                res_df=res_df,
                plots_folder=plot_path,
                metric_name=metric,
                name_add='_nice_features'
            )
