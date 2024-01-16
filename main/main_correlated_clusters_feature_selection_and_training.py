import argparse
import os
from datetime import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from jobs_params import predict_techability_retina_xor_after_feature_selection, \
    predict_techability_digits_after_feature_selection, predict_techability_xor_after_feature_selection
from logical_gates import LogicalGates
from networks_teachability_regression.regression_nn_learn import regression_lnn_learning
from networks_teachability_regression.regression_tree_feature_selection import RegressionTreeFeatureSelection
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim
from stractural_features_models.structural_features import NICE_FEATURES, NICE_FEATURES_NO_INV
from utils.regression_methods import get_list_features_names

parser = argparse.ArgumentParser()
parser.add_argument('--job_num', default=1)
parser.add_argument('--task', default='retina_xor')

args = parser.parse_args()
task = args.task
job_num = int(args.job_num)

features_list = NICE_FEATURES
all_options = False
if task == 'xor':
    param_dict = predict_techability_xor_after_feature_selection
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
    correlated_features_csv_name = selected_exp_names[task][name_].correlated_features_csv_name
elif task == 'retina_xor':
    param_dict = predict_techability_retina_xor_after_feature_selection
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
    train_path = selected_exp_names[task][name_].train_data_path
    test_path = selected_exp_names[task][name_].test_data_path
    correlated_features_csv_name = selected_exp_names[task][name_].correlated_features_csv_name
    features_list = NICE_FEATURES_NO_INV

elif task == 'digits':
    param_dict = predict_techability_digits_after_feature_selection
    dims = [64, 6, 6, 10]
    num_layers = len(dims) - 1
    task_params = DigitsByDim(
        start_dimensions=dims,
        num_layers=num_layers,
        by_epochs=False,
    )
    task_params.task_version_name = f'{task}_{num_layers}_layers'
    name_ = 'random'
    model_base_path_to_res = f"/lightgbm_feature_selection/{selected_exp_names[task][name_].feature_selection_folder}/masked_data_models/"
    data_path = "2023-11-27-18-30-52_masked_data_200_features.pkl"
    correlated_features_csv_name = selected_exp_names[task][name_].correlated_features_csv_name
    all_options = True
else:
    raise ValueError()

if __name__ == '__main__':
    num_features = selected_exp_names[task][name_].num_selected_features
    task_data_folder_name = 'train_test_data'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    #task_params.base_path = '/Volumes/noamaz/modularity'
    exp_name = f'{time_str}_{num_features}_features_{job_num}'

    out_folder = f'feature_correlation'
    out_dir = f"{task_params.teach_arch_base_path}/{out_folder}"
    out_features_dir = f"{out_dir}/{num_features}_features/"
    out_features_path = Path(out_features_dir)
    out_features_path.mkdir(exist_ok=True)
    models_folder_name = param_dict.get("model_path", "masked_data_models")
    res_output_path = param_dict.get("output_path", "teach_archs_regression_random_feature_selection_results")
    out_masked_data_dir = f"{out_features_dir}/{models_folder_name}"
    out_masked_data_path = Path(out_masked_data_dir)
    out_masked_data_path.mkdir(exist_ok=True)
    out_reg_res_dir = f"{out_features_dir}/{res_output_path}"
    out_reg_res_path = Path(out_reg_res_dir)
    out_reg_res_path.mkdir(exist_ok=True)

    correlated_features = pd.read_csv(f"{out_dir}/{correlated_features_csv_name}", index_col=0)

    exp_path = param_dict["exp_path"]
    exp_full_path = f'{task_params.base_path}/{exp_path}/{param_dict["exp_folder"]}'
    models_path = f'{exp_full_path}/{param_dict.get("model_path", "masked_data_models")}'
    learning_rate = param_dict['learning_rate']
    num_epochs = param_dict['num_epochs']
    batch_size = param_dict['batch_size']
    layers_sized = [406, 4096, 2048, 1024, 512, 64, 1]
    label_name = "mean_performance"

    for model_name in os.listdir(models_path):
        if '._' in model_name:
            continue
        if f'{num_features}_features' in model_name:
            with open(f"{models_path}/{model_name}", 'rb') as fp:
                masked_model = joblib.load(fp)
            break
    selected_feature_names = masked_model.get('selected_feature_names')

    if task == 'digits':
        with open(f'{task_params.teach_arch_base_path}/{model_base_path_to_res}/{data_path}', 'rb') as fp:
            data = joblib.load(fp)
        test_data = data['selected_train_data']
        train_data = data['selected_test_data']
        feature_names = data["selected_feature_names"]
    else:
        base_path_to_res = f"{task_params.teach_arch_base_path}/{task_data_folder_name}"
        with open(f'{base_path_to_res}/{test_path}', 'rb') as fp:
            test_data = joblib.load(fp)
        with open(f'{base_path_to_res}/{train_path}', 'rb') as fp:
            train_data = joblib.load(fp)
        feature_names = get_list_features_names(
            task_params=task_params,
            features_list=features_list,
        )
    selected_features_idxs = [
        correlated_features.columns.to_list().index(feature)
        for feature in selected_feature_names
    ]
    corr_vals = np.tril(correlated_features, k=-1).flatten()
    corr_vals = corr_vals[corr_vals != 0]
    corr_array = np.array(correlated_features)
    quentile = 0.95
    masked_ind = []
    for feature_idx in selected_features_idxs:
        ind = np.where(corr_array[feature_idx] >= np.quantile(corr_vals, quentile))[0]
        if all_options:
            if len(ind) == 2:
                selected = ind[ind != feature_idx][0]
            else:
                selected = ind[ind != feature_idx][job_num-1]
        else:
            if len(ind) > 1:
                selected = np.random.choice(ind[ind != feature_idx])
                while selected in masked_ind:
                    selected = np.random.choice(ind[ind != feature_idx])
            else:
                selected = np.argsort(corr_array[feature_idx])[-2]
        masked_ind.append(selected)
    mask = [
        False
        if i not in masked_ind
        else True
        for
        i in range(len(feature_names))
    ]
    mask_tens = torch.tensor(mask)
    selected_train_data = RegressionTreeFeatureSelection.mask_tensors(
        mask_tensor=mask_tens,
        data_tensors=train_data,
    )
    selected_test_data = RegressionTreeFeatureSelection.mask_tensors(
        mask_tensor=mask_tens,
        data_tensors=test_data,
    )
    new_selected_feature_names = np.array(feature_names)[mask].tolist()
    print(f'new_selected_feature_names: {new_selected_feature_names}')
    masked_data = {
        "num_features": num_features,
        "mask": mask_tens,
        "selected_train_data": selected_train_data,
        "selected_test_data": selected_test_data,
        "selected_feature_names": new_selected_feature_names,
    }
    model_path = f'{out_masked_data_path}/{exp_name}.pkl'
    with open(model_path, 'wb+') as fp:
        joblib.dump(masked_data, fp)
    regression_lnn_learning(
        layers_sized=layers_sized,
        train_data=selected_train_data,
        test_data=selected_test_data,
        epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        test_every=1,
        output_path=out_reg_res_dir,
        task=task,
        exp_name=exp_name,
        save_preds=True,
    )
