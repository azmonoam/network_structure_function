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

if __name__ == '__main__':
    # base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    base_path = '/home/labs/schneidmann/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--task', default='retina')
    parser.add_argument('--regresor', default='lightgbm')
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
    learning_rate = param_dict['learning_rate']
    num_epochs = param_dict['num_epochs']
    batch_size = param_dict['batch_size']
    layers_sized = [406, 4096, 2048, 1024, 512, 64, 1]
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

    sufix = f'_meta_only_{num_features}_features{param_dict.get("sufix", "")}{model_name}'

    regression_lnn_learning(
        layers_sized=layers_sized,
        train_data=train_data,
        test_data=test_data,
        epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        test_every=1,
        sufix=sufix,
        output_path=output_path,
        task=task,
    )
    print('end')
