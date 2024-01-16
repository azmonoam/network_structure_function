import argparse
import itertools
import os
import random
from datetime import datetime as dt
from pathlib import Path

import joblib
import numpy as np
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
parser.add_argument('--task', default='digits')
parser.add_argument('--job_num', default=1)

args = parser.parse_args()
task = args.task
job_number = int(args.job_num)

features_list = NICE_FEATURES
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
        'num_involved_neurons_in_paths_per_input_neuron',
    ],
    0: [
        'out_connections_per_layer',
        'in_connections_per_layer',
    ]
}
all_options = True
use_orig_features = True
name_add = ''
if task == 'xor':
    param_dict = predict_techability_xor_after_feature_selection
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
elif task == 'retina_xor':
    all_options = False
    param_dict = predict_techability_retina_xor_after_feature_selection
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
    train_path = selected_exp_names[task][name_].train_data_path
    test_path = selected_exp_names[task][name_].test_data_path
    features_list = NICE_FEATURES_NO_INV
elif task == 'digits':
    param_dict = predict_techability_digits_after_feature_selection
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
    raise ValueError()

if __name__ == '__main__':
    features_to_remove = {
        'dimensions_0',
        f'dimensions_{num_layers}',
        'num_layers',
        'num_neurons',
        f'out_connections_per_layer_({num_layers}_ 0) '
    }
    out_degree = {
        f'out_connections_per_layer_({num_layers}_ {i}) '
        for i in range(selected_exp_names[task][name_].output_size)
    }
    in_degree = {
        f'in_connections_per_layer_(0_ {i}) '
        for i in range(selected_exp_names[task][name_].input_size)
    }
    features_to_remove = features_to_remove.union(out_degree)
    features_to_remove = features_to_remove.union(in_degree)

    num_features = selected_exp_names[task][name_].num_selected_features

    task_data_folder_name = 'train_test_data'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    #task_params.base_path = '/Volumes/noamaz/modularity'
    exp_name = f'{time_str}_{num_features}_features_{job_number}'

    out_folder = f'feature_globality'
    out_dir = f"{task_params.teach_arch_base_path}/{out_folder}"
    out_features_dir = f"{out_dir}/{num_features}_features"
    out_features_path = Path(out_features_dir)
    out_features_path.mkdir(exist_ok=True)

    exp_path = param_dict["exp_path"]
    exp_full_path = f'{task_params.base_path}/{exp_path}/{param_dict["exp_folder"]}'
    models_folder_name = param_dict.get("model_path", "masked_data_models")
    res_output_path = param_dict.get("output_path", "teach_archs_regression_random_feature_selection_results")
    models_path = f'{exp_full_path}/{models_folder_name}'
    learning_rate = param_dict['learning_rate']
    num_epochs = param_dict['num_epochs']
    batch_size = param_dict['batch_size']
    layers_sized = [406, 4096, 2048, 1024, 512, 64, 1]
    label_name = "mean_performance"

    out_features_path = Path(out_features_dir)
    out_features_path.mkdir(exist_ok=True)
    out_masked_data_dir = f"{out_features_dir}/{models_folder_name}"
    out_masked_data_path = Path(out_masked_data_dir)
    out_masked_data_path.mkdir(exist_ok=True)
    out_reg_res_dir = f"{out_features_dir}/{res_output_path}"
    out_reg_res_path = Path(out_reg_res_dir)
    out_reg_res_path.mkdir(exist_ok=True)

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
        all_feature_names = data["selected_feature_names"]
    else:
        base_path_to_res = f"{task_params.teach_arch_base_path}/{task_data_folder_name}"
        with open(f'{base_path_to_res}/{test_path}', 'rb') as fp:
            test_data = joblib.load(fp)
        with open(f'{base_path_to_res}/{train_path}', 'rb') as fp:
            train_data = joblib.load(fp)

        all_feature_names = get_list_features_names(
            task_params=task_params,
            features_list=features_list,
        )
    allowed_features = [
        feature
        for feature in all_feature_names if feature not in features_to_remove
    ]
    globality_level_full_feature_names = {
        k: []
        for k in globality_level.keys()
    }
    for feature_full_name in allowed_features:
        for globality_idx, features_list in globality_level.items():
            for f in features_list:
                if f == 'connectivity_ratio' and f != feature_full_name:
                    continue
                if f in feature_full_name:
                    globality_level_full_feature_names[globality_idx].append(feature_full_name)

    glob_to_sample = {
        globality_idx: 0
        for globality_idx in globality_level_full_feature_names.keys()
    }
    for selected_feature in selected_feature_names:
        for globality_idx, features_list in globality_level_full_feature_names.items():
            if selected_feature in features_list:
                glob_to_sample[globality_idx] += 1
                if not use_orig_features:
                    features_list.pop(features_list.index(selected_feature))
    if all_options:
        combs = []
        for globality_idx, k in glob_to_sample.items():
            if k == 0:
                continue
            combs.append(list(itertools.combinations(globality_level_full_feature_names[globality_idx], k)))
        all_combs = [list(itertools.chain(*c)) for c in itertools.product(*combs)]
        if use_orig_features:
            all_combs.pop([all_combs.index(list(x)) for x in list(itertools.permutations(selected_feature_names)) if list(x) in all_combs][0])
        random.seed(1994)
        random.shuffle(all_combs)
        globality_fitted_features = all_combs[job_number-1]
    else:
        globality_fitted_features = selected_feature_names
        while globality_fitted_features == selected_feature_names:
            globality_fitted_features = []
            for globality_idx, k in glob_to_sample.items():
                if k == 0:
                    continue
                globality_fitted_features += random.choices(globality_level_full_feature_names[globality_idx], k=k)

    masked_ind = [
        all_feature_names.index(f) for f in globality_fitted_features
    ]
    mask = [
        False
        if i not in masked_ind
        else True
        for
        i in range(len(all_feature_names))
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
    masked_data = {
        "num_features": num_features,
        "mask": mask_tens,
        "selected_train_data": selected_train_data,
        "selected_test_data": selected_test_data,
        "selected_feature_names": np.array(all_feature_names)[mask].tolist(),
    }
    model_path = f'{out_masked_data_dir}/{exp_name}.pkl'
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
