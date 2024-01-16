import os
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd

from analsis.analsis_utils.plot_utils import COLORS


def get_all_num_features_results(
        res_folder: str,
        start_idx: int = 50,
        end_idx: int = 10000,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    res_df = pd.DataFrame()
    all_results_dict = {}
    for i, csv_name in enumerate(os.listdir(res_folder)):
        if csv_name.find('feature_selection') != -1:
            continue
        if ".csv" not in csv_name:
            continue
        results_meta_only = pd.read_csv(f"{res_folder}/{csv_name}")
        results_meta_only['log loss'] = np.log2(results_meta_only['losses'])
        num_features = int(csv_name.split('_')[-2])
        res_dict = {
            'num_features': num_features,
            'mean_train_r2': results_meta_only['r2s train'][start_idx:end_idx].mean(),
            'mean_test_r2': results_meta_only['r2s test'][start_idx:end_idx].mean(),
            'max_train_r2': results_meta_only['r2s train'][start_idx:end_idx].max(),
            'max_test_r2': results_meta_only['r2s test'][start_idx:end_idx].max(),
        }
        res_df = pd.concat([res_df, pd.DataFrame(res_dict, index=[0], )], ignore_index=True)
        all_results_dict[num_features] = results_meta_only

    res_df = res_df
    return all_results_dict, res_df


def get_all_num_features_results_and_desired_external_exp(
        csv_to_compere_path: str,
        comparison_name: str,
        res_folders: List[str],
        regressors: List[str],
        start_idx: int = 50,
        end_idx: int = 10000,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    res_df = pd.DataFrame()
    all_results_dict = {}
    for res_folder, regressor in zip(res_folders, regressors):
        regressor_all_results_dict, regressor_res_df = get_all_num_features_results(
            res_folder=res_folder,
            start_idx=start_idx,
            end_idx=end_idx
        )
        for key, val in regressor_all_results_dict.items():
            all_results_dict[f"{key}_{regressor}"] = val

        regressor_res_df['name'] = regressor_res_df['num_features'].astype(str) + f'_{regressor}'
        res_df = pd.concat([res_df, pd.DataFrame(regressor_res_df, index=[0], )], ignore_index=True)

    results_to_compere = pd.read_csv(csv_to_compere_path)
    results_to_compere['log loss'] = np.log2(results_to_compere['losses'])
    results_to_compere_dict = {
        'name': comparison_name,
        'mean_train_r2': results_to_compere['r2s train'][start_idx:end_idx].mean(),
        'mean_test_r2': results_to_compere['r2s test'][start_idx:end_idx].mean(),
        'max_train_r2': results_to_compere['r2s train'][start_idx:end_idx].max(),
        'max_test_r2': results_to_compere['r2s test'][start_idx:end_idx].max(),
    }
    res_df = pd.concat([res_df, pd.DataFrame(results_to_compere_dict, index=[0], )], ignore_index=True)
    all_results_dict[comparison_name] = results_to_compere

    return all_results_dict, res_df


def get_all_num_features_models_masks(
        models_folder: str,
) -> List[Tuple[int, np.ndarray]]:
    all_models_masks = []
    for model_path in os.listdir(models_folder):
        if model_path[0:1] == '._':
            continue
        with open(f'{models_folder}/{model_path}', 'rb') as fp:
            model = joblib.load(fp)
        all_models_masks.append(
            (
                model.get('num_features'),
                np.array(model.get('mask')),
            )
        )
    return all_models_masks


def prepare_data_of_used_features(
        used_features_dict: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    global_features = ['modularity',
                       'entropy',
                       'normed_entropy',
                       'max_connectivity_between_layers_per_layer_3',
                       'total_connectivity_ratio_between_layers_0',
                       'total_connectivity_ratio_between_layers_3',
                       'density',
                       'motifs_0',
                       'motifs_1',
                       'motifs_2',
                       ]
    features = []
    num_uses = []
    bar_colors = []
    bar_labels = []
    for k, v in used_features_dict.items():
        features.append(k)
        num_uses.append(len(v))
        if k in global_features:
            bar_colors.append(COLORS[0])
            bar_labels.append('global')
        else:
            bar_colors.append(COLORS[1])
            bar_labels.append('local')
    sorted_vals = [(x, y, z, t) for x, y, z, t in sorted(zip(num_uses, features, bar_colors, bar_labels), reverse=True)]
    features = []
    num_uses = []
    bar_colors = []
    bar_labels = []
    for num_uses_, feature_, bar_color_, bar_label_ in sorted_vals:
        num_uses.append(num_uses_)
        features.append(feature_)
        bar_colors.append(bar_color_)
        bar_labels.append(bar_label_)
    return num_uses, features, bar_colors, bar_labels


def collect_num_uses_different_exp(
        base_path: str,
        csvs_mapping: Dict[str, str],
        num_features: int,
) -> Dict[str, List[str]]:
    used_features_dict = {}
    for algo_name, csv_path in csvs_mapping.items():
        selected_features_df = pd.read_csv(f"{base_path}/{csv_path}").drop("Unnamed: 0", axis=1).rename(
            columns={'connectivity_ratio': 'density'})
        selected_features = selected_features_df[selected_features_df.sum(axis=1) == num_features]
        selected_feature_names = selected_features[selected_features == 1].dropna(axis=1).columns
        for feature_name in selected_feature_names:
            if feature_name not in used_features_dict:
                used_features_dict[feature_name] = []
            used_features_dict[feature_name].append(algo_name)
    return used_features_dict
