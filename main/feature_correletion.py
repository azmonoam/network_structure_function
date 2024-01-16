import argparse
from datetime import datetime as dt
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from logical_gates import LogicalGates
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim
from stractural_features_models.structural_features import NICE_FEATURES, NICE_FEATURES_NO_INV
from utils.regression_methods import get_list_features_names

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='retina_xor')

args = parser.parse_args()
task = args.task

features_list = NICE_FEATURES
name_add = ''
if task == 'xor':
    col = 'Reds'
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
    plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_global_name}_multi_arch/{task_params.task_version_name}'

elif task == 'retina_xor':
    col = 'Blues'
    dims = [6, 5, 2, 2]
    num_layers = len(dims) - 1
    task_params = RetinaByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
        task_base_folder_name='retina_xor',
        rule=LogicalGates.XOR,
    )
    random_archs = True
    if random_archs:
        task_params.task_version_name = f'retina_{num_layers}_layers'
        name_ = 'random'
    else:
        name_ = tuple([d for d in dims[1:-1]])
    features_list = NICE_FEATURES_NO_INV
    train_path = selected_exp_names[task][name_].train_data_path
    test_path = selected_exp_names[task][name_].test_data_path
    plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_base_folder_name}_multi_arch/{task_params.task_version_name}'
elif task == 'digits':
    col = 'Greens'
    dims = [64, 6, 6, 10]
    num_layers = len(dims) - 1
    task_params = DigitsByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    random_archs = True
    if random_archs:
        task_params.task_version_name = f'{task}_{num_layers}_layers'
        name_ = 'random'
        model_base_path_to_res = f"/lightgbm_feature_selection/{selected_exp_names[task][name_].feature_selection_folder}/masked_data_models/"
        data_path = "2023-11-27-18-30-52_masked_data_200_features.pkl"
    else:
        name_ = tuple([d for d in dims[1:-1]])
    exp_folder_name_addition = ''
    plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_global_name}_multi_arch/{task_params.task_version_name}'

else:
    raise ValueError()

if __name__ == '__main__':
    task_data_folder_name = 'train_test_data'
    local_base_path = '/home/labs/schneidmann/noamaz/network_modularity'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = task_params.base_path
    task_params.base_path = '/Volumes/noamaz/modularity'

    out_folder = f'feature_correlation'

    base_path_to_res = f"{task_params.teach_arch_base_path}/{task_data_folder_name}"
    out_dir = f"{task_params.teach_arch_base_path}/{out_folder}"
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    if task == 'digits':
        with open(f'{task_params.teach_arch_base_path}/{model_base_path_to_res}/{data_path}', 'rb') as fp:
            data = joblib.load(fp)
        test_data = data['selected_train_data']
        train_data = data['selected_test_data']
        feature_names = data["selected_feature_names"]
    else:
        with open(f'{base_path_to_res}/{test_path}', 'rb') as fp:
            test_data = joblib.load(fp)
        with open(f'{base_path_to_res}/{train_path}', 'rb') as fp:
            train_data = joblib.load(fp)
        feature_names = get_list_features_names(
            task_params=task_params,
            features_list=features_list,
        )
    data_all = [d.tolist() for d, l in train_data + test_data]
    np_data = np.array(data_all)
    num_features = np.shape(np_data)[1]

    const_inds = []
    for i in range(num_features):
        if len(set(np_data[:, i])) == 1:
            const_inds.append(i)
    np_data_no_consts = np.delete(np_data, const_inds, 1)
    feature_names = [f for i, f in enumerate(feature_names) if i not in const_inds]
    num_features -= len(const_inds)
    corrr = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            corrr[i, j] = abs(pearsonr(np_data_no_consts[:, i], np_data_no_consts[:, j]).statistic)
            corrr[j, i] = corrr[i, j]
    corr_df = pd.DataFrame(corrr)
    corr_df.columns = feature_names
    corr_df.index = feature_names
    all_data_df = pd.DataFrame(np_data_no_consts)
    all_data_df.columns = feature_names
    all_data_df.to_csv(f"{out_dir}/{time_str}_all_data.csv")
    corr_df.to_csv(f"{out_dir}/{time_str}_feature_correlation.csv")
    c = sns.color_palette(col, as_cmap=True)
    ax = sns.heatmap(corrr, linewidth=0.5, cmap=c)
    plt.title(f"{task} - feature correlation (abs)")
    plt.savefig(
        f"{plot_path}/{time_str}_{task}_feature_correlation{name_add}.png")
    plt.show()
    print('a')
