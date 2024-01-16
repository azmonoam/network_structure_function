from datetime import datetime as dt
from typing import List

import joblib
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from logical_gates import LogicalGates
from networks_teachability_regression.regression_tree_learn import tree_regression
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.xor.xor_by_dim import XoraByDim
from parameters.digits.digits_by_dim import DigitsByDim

from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from utils.set_up_population_utils import get_organism_by_connectivity_ratio

COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']
task = 'digits'
num_layers = 2
local_base_path = '/Users/noamazmon/PycharmProjects/network_modularity'


def _get_feature_mapping(
        task_params,
) -> List[str]:
    organism = get_organism_by_connectivity_ratio(
        task_params=task_params,
        connectivity_ratio=0.5,
    )
    structural_features_calculator = CalcStructuralFeatures(
        organism=organism,
    )
    organism = structural_features_calculator.calc_structural_features()

    return list(organism.structural_features.get_class_constant_features(
        organism.layer_neuron_idx_mapping
    ).keys())


if task == 'retina':
    task_data_path = f'retina/retina_3_layers_3_4/'
    out_folder = "lightgbm_regression_tree"
    dims = [6, 3, 4, 2]
    task_params = RetinaByDim(
        start_dimensions=dims,
        num_layers=num_layers,
    )
    local_out_folder = f'{local_base_path}/plots/retina_multi_archs/{num_layers}_layers/{out_folder}'
elif task == 'retina_xor':
    size_folder = 'retina_3_layers_3_4'
    task_data_path = f'retina_xor/{size_folder}/'
    out_folder = "lightgbm_regression_tree"
    dims = [6, 3, 4, 2]
    task_params = RetinaByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
        task_base_folder_name='retina_xor',
        rule=LogicalGates.XOR,
    )
    local_out_folder = f'{local_base_path}/plots/retina_xor_multi_arch/{size_folder}/{out_folder}'
    train_path = 'retina_xor_train_2023-09-04-11-44-55_adj_False_meta_True_10k_ep.pkl'
    test_path = 'retina_xor_test_2023-09-04-11-44-55_adj_False_meta_True_10k_ep.pkl'

elif task == 'xor':
    size_folder = 'xor_4_layers_6_5_3'
    task_data_path = f'xor/{size_folder}/'
    out_folder = "lightgbm_regression_tree"
    dims = [6, 6, 5, 3, 2]
    task_params = XoraByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    local_out_folder = f'{local_base_path}/plots/xor_multi_arch/{size_folder}/{out_folder}'


elif task == 'digits':
    size_folder = 'digits_2_layers_16'
    task_data_path = f'digits/{size_folder}/'
    out_folder = "lightgbm_regression_tree"
    dims = [64, 16, 2]
    task_params = DigitsByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    local_out_folder = f'{local_base_path}/plots/digits_multi_arch/{size_folder}/{out_folder}'
    train_path = 'digits_train_2023-09-07-11-13-06_adj_False_meta_True_10k_ep.pkl'
    test_path = 'digits_test_2023-09-07-11-13-06_adj_False_meta_True_10k_ep.pkl'

else:
    raise ValueError

if __name__ == '__main__':
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = '/Volumes/noamaz/modularity'
    # base_path = '/home/labs/schneidmann/noamaz/modularity'
    base_path_to_res = f"{base_path}/{task_data_path}/train_test_data"


    num_feature = 20
    n_estimators = 500
    learning_rate = 0.001
    subsample_for_bin = 2000
    mapping = _get_feature_mapping(
        task_params=task_params,
    )

    model, train_r2, test_r2 = tree_regression(
        train_path=f'{base_path_to_res}/{train_path}',
        test_path=f'{base_path_to_res}/{test_path}',
        n_estimators=n_estimators,
    )
    res = {
        "model": model,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mse_loss_train": model.evals_result_['train'],
        "mse_loss_test": model.evals_result_['test'],
    }
    with open(f"{base_path}/{task_data_path}/{out_folder}/{time_str}_lightgbm_regression_results.pkl", 'wb+') as fp:
        joblib.dump(res, fp)
    lightgbm.plot_metric(model, title=f'{task} mean squared error during training',
                         dataset_names=['train', 'test'],
                         ylabel='Mean squared error')
    plt.savefig(
        f"{local_out_folder}/{time_str}_{task}_mean_squared_error_lightgbm_training.png")
    plt.show()

    feature_importance = pd.DataFrame(
        sorted(zip(model.feature_importances_, mapping), reverse=True),
        columns=['Value', 'Feature'],
    )
    plt.figure()
    sns.barplot(x="Value", y='Feature', data=feature_importance.iloc[:num_feature],
                palette=sns.color_palette("flare", n_colors=num_feature)
                )
    plt.title(f'{task} - LightGBM Features')
    plt.tight_layout()
    plt.savefig(
        f"{local_out_folder}/{time_str}_{task}_feature_importance_top_{num_feature}_lightgbm_training.png")
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6))
    bar_size = 0.25
    padding = 0.15
    y_locs = np.arange(2 * (bar_size * 3 + padding))
    ax1.plot(list(range(len(model.evals_result_['test']['l2'])))[-30:], model.evals_result_['test']['l2'][-30:],
             c=COLORS[0])
    ax2.plot(list(range(len(model.evals_result_['train']['l2'])))[-30:], model.evals_result_['train']['l2'][-30:],
             c=COLORS[0])
    ax3.barh(y_locs + (0 * bar_size), [train_r2, test_r2], height=bar_size, color=COLORS[0], )

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE loss- Test')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE loss - Train')
    ax3.set(
        yticks=y_locs,
        yticklabels=['Train', 'Test'],
        ylim=[0 - padding, len(y_locs) - padding],
    )
    ax3.set_xlabel('r2 score')
    ax1.set_title(f'{task} - Prediction of teachability mean performance')
    plt.savefig(f"{local_out_folder}/{time_str}_{task}_data_type_comparison_lightgbm_regression.png")
    plt.show()
