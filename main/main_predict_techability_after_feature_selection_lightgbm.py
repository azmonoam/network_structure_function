import argparse

import joblib
import pandas as pd

from jobs_params import (
    main_predict_techability_xor_lightgbm_feature_selection,
    main_predict_techability_retina_lightgbm_feature_selection,
    main_predict_techability_retina_xgboost_feature_selection,
    main_predict_techability_xor_xgboost_feature_selection,
    main_predict_techability_retina_ind_feature_selection,
    main_predict_techability_xor_ind_feature_selection,
    main_predict_techability_digits_lightgbm_feature_selection,
    main_predict_techability_digit_ind_feature_selection,
)
from networks_teachability_regression.regression_nn_learn import regression_lnn_learning
from networks_teachability_regression.regression_tree_learn import tree_regression
from datetime import datetime as dt

if __name__ == '__main__':
    base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=11)
    parser.add_argument('--task', default='digits')
    parser.add_argument('--regresor', default='none')
    args = parser.parse_args()
    job_num = int(args.job_num)
    task = args.task
    regresor = args.regresor
    if regresor == 'lightgbm':
        if task == 'retina':
            param_dict = main_predict_techability_retina_lightgbm_feature_selection
        elif task == 'xor':
            param_dict = main_predict_techability_xor_lightgbm_feature_selection
        elif task == 'digits':
            param_dict = main_predict_techability_digits_lightgbm_feature_selection
    elif regresor == 'xgboost':
        if task == 'retina':
            param_dict = main_predict_techability_retina_xgboost_feature_selection
        elif task == 'xor':
            param_dict = main_predict_techability_xor_xgboost_feature_selection
    elif regresor == 'none':
        if task == 'retina':
            param_dict = main_predict_techability_retina_ind_feature_selection
        elif task == 'xor':
            param_dict = main_predict_techability_xor_ind_feature_selection
        elif task == 'digits':
            param_dict = main_predict_techability_digit_ind_feature_selection
    exp_path = param_dict["exp_path"]
    output_path = f'{base_path}/{exp_path}/{param_dict["output_path"]}'
    label_name = "mean_performance"

    print('start')
    if regresor == 'none':
        model_path = f"{base_path}/{exp_path}/{param_dict[job_num]['model_path']}"
        model_name = f"_{param_dict[job_num]['model_name']}"
    else:
        csv_path = f'{base_path}/{exp_path}/{param_dict["csv_path"]}'
        model_file_name = f"{pd.read_csv(csv_path)['model_name'].iloc[job_num - 1].split('/')[-1]}"
        model_path = f'{base_path}/{exp_path}/masked_data_models/{model_file_name}'
        model_name = ''
    with open(model_path, 'rb') as fp:
        masked_model = joblib.load(fp)
    train_data = masked_model.get('selected_train_data')
    test_data = masked_model.get('selected_test_data')
    num_features = masked_model.get('num_features')


    model, train_r2, test_r2 = tree_regression(
        train_data=train_data,
        test_data=test_data,
    )
    test_mse = model.evals_result_['test']['l2']
    train_mse = model.evals_result_['test']['l2']
    red_df = pd.DataFrame(
        {
            'r2s train': [train_r2]*len(test_mse),
            'r2s test': [test_r2]*len(test_mse),
            'loss_test': test_mse,
            'loss_train': train_mse,
        }
    )
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    red_df.to_csv(f"{output_path}/{task}_{time_str}_lightgbm_{model_name}.csv", )
    print('end')
