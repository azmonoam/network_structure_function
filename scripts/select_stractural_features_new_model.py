from datetime import datetime as dt
from typing import List

import joblib
import torch
import numpy as np


from networks_teachability_regression.regression_tree_feature_selection import RegressionTreeFeatureSelection
from train_test_data_selector import DataSelector
from parameters.retina.retina_b_params import RetinaBParams


def _find_desired_structural_features_idxs(
        desired_structural_features: List[str],
        structural_features_names_vec: List[str],
) -> torch.Tensor:
    idxs = torch.zeros(len(structural_features_names_vec))
    for i, feature_name in enumerate(structural_features_names_vec):
        for desired_structural_feature_name in desired_structural_features:
            if desired_structural_feature_name in ['connectivity_ratio',
                                                   'entropy'] and feature_name != desired_structural_feature_name:
                continue
            if desired_structural_feature_name in feature_name:
                idxs[i] = int(1)
                break
    return idxs.to(torch.bool)


if __name__ == '__main__':
    local_base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = '/Volumes/noamaz/modularity'
    # base_path = '/home/labs/schneidmann/noamaz/modularity'

    task = 'retina'
    with_motifs = True
    no_modularity = True
    params = RetinaBParams()

    base_path_to_res = f"{base_path}/{task}/{params.task_version_name}/train_test_data"
    models_folder = f"{base_path}/{task}/{params.task_version_name}/feature_selection/masked_data_and_models"

    train_path = 'retina_train_2023-08-09-11-14-03_adj_False_meta_True_with_motifs.pkl'
    test_path = 'retina_test_2023-08-09-11-14-03_adj_False_meta_True_with_motifs.pkl'

    desired_structural_features = [
        "motifs",
        "connectivity_ratio",
        "entropy"
    ]
    model_name = f'{time_str}_masked_data'
    for feature in desired_structural_features:
        model_name += f"_{feature}"
    model_name += '.pkl'

    structural_features_full_name_vec = params.structural_features_full_name_vec_with_motifs

    with open(f'{base_path_to_res}/{test_path}', 'rb') as fp:
        test_data = joblib.load(fp)
    with open(f'{base_path_to_res}/{train_path}', 'rb') as fp:
        train_data = joblib.load(fp)
    idxs = _find_desired_structural_features_idxs(
        structural_features_names_vec=structural_features_full_name_vec,
        desired_structural_features=desired_structural_features,
    )
    masked_test_data = RegressionTreeFeatureSelection.mask_tensors(
        mask_tensor=idxs,
        data_tensors=test_data,
    )
    masked_train_data = RegressionTreeFeatureSelection.mask_tensors(
        mask_tensor=idxs,
        data_tensors=train_data,
    )
    masked_data = {
        'num_features': int(sum(idxs)),
        "mask": idxs,
        "selected_train_data": masked_train_data,
        "selected_test_data": masked_test_data,
        "selected_feature_names": np.array(structural_features_full_name_vec)[idxs].tolist(),
    }
    model_path = f'{models_folder}/{model_name}'
    with open(model_path, 'wb+') as fp:
        joblib.dump(masked_data, fp)
    print(model_path)
