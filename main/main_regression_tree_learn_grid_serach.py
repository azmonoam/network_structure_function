import argparse
from datetime import datetime as dt
from pathlib import Path
from stractural_features_models.structural_features import CONSTANT_FEATURES_PER_ARCH
from jobs_params import regression_tree_num_features
from logical_gates import LogicalGates
from networks_teachability_regression.regression_tree_feature_selection_lightgbm import \
    LightGBMRegressionTreeFeatureSelection
from networks_teachability_regression.regression_tree_feature_selection_xboost import \
    XGBoostRegressionTreeFeatureSelection
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.xor.xor_by_dim import XoraByDim
import lightgbm
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='retina_xor')
parser.add_argument('--regressor', default='lightgbm')
parser.add_argument('--n_threads', default=1)
parser.add_argument('--job_num', default=1)
parser.add_argument('--no_modularity', default=1)
parser.add_argument('--step', default=2)

args = parser.parse_args()
task = args.task
n_threads = int(args.n_threads)
job_number = int(args.job_num)
regressor = args.regressor
no_modularity = bool(int(args.no_modularity))

step = int(args.step)
n_estimators = 500
features_to_drop = None
features_list = None
if no_modularity:
    features_to_drop = ['modularity']

if task == 'xor':
    dims = [6, 6, 5, 3, 2]
    num_layers = len(dims) - 1
    task_params = XoraByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    train_path = 'xor_train_2023-09-07-13-57-56_adj_False_meta_True_10k_ep.pkl',
    test_path = 'xor_train_2023-09-07-13-57-56_adj_False_meta_True_10k_ep.pkl',

elif task == 'retina_xor':
    dims = [6, 3, 4, 2]
    num_layers = len(dims) - 1
    task_params = RetinaByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
        task_base_folder_name='retina_xor',
        rule=LogicalGates.XOR,
    )
    train_path = 'retina_xor_train_2023-09-04-11-44-55_adj_False_meta_True_10k_ep_no_mod.pkl'
    test_path = 'retina_xor_test_2023-09-04-11-44-55_adj_False_meta_True_10k_ep_no_mod.pkl'
    train_path = 'retina_xor_train_2023-09-09-10-20-18_adj_False_meta_True_10k_ep_const_features_per_arch.pkl'
    test_path = 'retina_xor_test_2023-09-09-10-20-18_adj_False_meta_True_10k_ep_const_features_per_arch.pkl'
    features_to_drop = None
    features_list = CONSTANT_FEATURES_PER_ARCH

elif task == 'digits':
    dims = [64, 16, 2]
    num_layers = len(dims) - 1
    task_params = DigitsByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    train_path = 'digits_train_2023-09-07-11-13-06_adj_False_meta_True_10k_ep.pkl'
    test_path = 'digits_test_2023-09-07-11-13-06_adj_False_meta_True_10k_ep.pkl'

else:
    raise ValueError()

if __name__ == '__main__':
    plot_out = False
    task_data_folder_name = 'train_test_data'
    local_base_path = '/home/labs/schneidmann/noamaz/network_modularity'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = task_params.base_path
    task_params.base_path = '/Volumes/noamaz/modularity'

    plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_base_folder_name}_multi_arch/{task_params.task_version_name}'

    out_folder = f'{regressor}_feature_selection'

    base_path_to_res = f"{task_params.teach_arch_base_path}/{task_data_folder_name}"
    out_path = f"{task_params.teach_arch_base_path}/{out_folder}"
    if regressor == 'lightgbm':
        feature_selection_regression_tree = LightGBMRegressionTreeFeatureSelection(
            base_path_to_res=base_path_to_res,
            test_path=test_path,
            train_path=train_path,
            out_folder=out_folder,
            out_path=out_path,
            time_str=time_str,
            task_params=task_params,
            n_threads=n_threads,
            n_estimators=n_estimators,
            features_to_drop=features_to_drop,
            features_list=features_list,
        )
    elif regressor == 'xgboost':
        record_folder = Path(f"{out_path}")
        feature_selection_regression_tree = XGBoostRegressionTreeFeatureSelection(
            base_path_to_res=base_path_to_res,
            test_path=test_path,
            train_path=train_path,
            out_folder=out_folder,
            out_path=out_path,
            time_str=time_str,
            task_params=task_params,
            n_threads=n_threads,
            features_to_drop=features_to_drop,
            features_list=features_list,
        )
    else:
        raise ValueError

    test_inputs, test_labels, train_inputs, train_labels = feature_selection_regression_tree._get_train_test_data()
    grid_param_space = {
        'num_leaves':[5,10,20,30,50],
        'min_child_samples': [100,200,500],
        'min_child_weight': [1e-5,  1e-1, 1, 1e1,  1e4],
        'learning_rate': [0.1,0.001,0.01],
        'reg_alpha': [0, 1e-1, 1, 5, 100],
        'reg_lambda': [0, 1e-1, 1, 5, 100]}
    model = lightgbm.LGBMRegressor(
        n_estimators=2000,
        force_col_wise=True,
    )
    GS_res = GridSearchCV(model, grid_param_space).fit(
        train_inputs,
        train_labels,
        eval_set=[(test_inputs, test_labels),
                  (train_inputs, train_labels)],
        eval_metric=feature_selection_regression_tree.eval_metric,
        eval_names=feature_selection_regression_tree.eval_names,
        feature_name=feature_selection_regression_tree.feature_names,
    )
    model = GS_res.best_estimator_
    best_params = GS_res.best_params_
