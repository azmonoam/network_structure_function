from datetime import datetime as dt

import pandas as pd

from networks_teachability_regression.regression_tree_feature_selection_lightgbm import \
    LightGBMRegressionTreeFeatureSelection
from networks_teachability_regression.regression_tree_feature_selection_xboost import \
    XGBoostRegressionTreeFeatureSelection
from parameters.retina_parameters import retina_structural_features_full_name_vec
from parameters.xor_parameters import xor_structural_features_full_name_vec

if __name__ == '__main__':
    local_base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = '/Volumes/noamaz/modularity'


    task = 'retina'
    regressor = "lightgbm"

    if task == "retina":
        feature_names = [a.replace(', ', '_') for a in retina_structural_features_full_name_vec]
        base_path_to_res = f"{base_path}/teach_archs/retina/retina_train_test_data"
        out_folder = f'retina_{regressor}_feature_selection'
        out_path = f"{base_path}/teach_archs/retina/{out_folder}"
        train_path = 'retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl'
        test_path = 'retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl'

    else:
        feature_names = [a.replace(', ', '_') for a in xor_structural_features_full_name_vec]
        base_path_to_res = f"{base_path}/teach_archs/xors/xor_train_test_data"
        out_folder = f'xor_{regressor}_feature_selection'
        out_path = f"{base_path}/teach_archs/xors/{out_folder}"
        train_path = 'xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl'
        test_path = 'xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl'

    exp_name = "exp_2023-04-25-12-22-31"

    feature_selection_csv_path = f"{out_path}/{exp_name}/2023-04-25-12-22-31_feature_selection.csv"
    used_features_csv_path = f"{out_path}/{exp_name}/2023-04-25-12-22-31_used_features.csv"

    if regressor == "lightgbm":
        feature_selection_regression_tree = LightGBMRegressionTreeFeatureSelection(
            base_path_to_res=base_path_to_res,
            test_path=test_path,
            train_path=train_path,
            feature_names=feature_names,
            out_folder=out_folder,
            out_path=out_path,
            time_str=time_str,
            task=task,
        )
    else:
        feature_selection_regression_tree = XGBoostRegressionTreeFeatureSelection(
            base_path_to_res=base_path_to_res,
            test_path=test_path,
            train_path=train_path,
            feature_names=feature_names,
            out_folder=out_folder,
            out_path=out_path,
            time_str=time_str,
            task=task,
        )
    feature_selection_regression_tree.models_folder = f"{out_path}/{exp_name}/masked_data_models"
    res_df_for_number_of_features, models_df_for_number_of_features = feature_selection_regression_tree.train_single_num_features(
        num_features=1
    )
    res_df = pd.read_csv(feature_selection_csv_path).drop("Unnamed: 0", axis=1).rename(
        columns={'model_path': "model_name"})
    models_df = pd.read_csv(used_features_csv_path).drop("Unnamed: 0", axis=1).rename(
        columns={'model_path': "model_name"})
    res_df = pd.concat([res_df, res_df_for_number_of_features], ignore_index=True)
    models_df = pd.concat([models_df, models_df_for_number_of_features], ignore_index=True)
    res_df = res_df.sort_values('num_features')

    res_df.to_csv(f"{out_path}/{exp_name}/{time_str}_feature_selection.csv")
    models_df.to_csv(f"{out_path}/{exp_name}/{time_str}_used_features.csv")
