import argparse
import os

import joblib

from jobs_params import (
    predict_techability_retina_xor_after_feature_selection,
    predict_techability_xor_after_feature_selection,
    predict_techability_digits_after_feature_selection,
)
from networks_teachability_regression.regression_nn_learn import regression_lnn_learning
from pathlib import Path
from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    #base_path = "/Volumes/noamaz/modularity/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=1)
    parser.add_argument('--task', default='retina_xor')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task = args.task

    if task == 'retina_xor':
        param_dict = predict_techability_retina_xor_after_feature_selection
    elif task == 'xor':
        param_dict = predict_techability_xor_after_feature_selection
    elif task == 'digits':
        param_dict = predict_techability_digits_after_feature_selection
    else:
        raise ValueError
    selected_params = selected_exp_names[task]['random']
    folder_path = f'{task}/{selected_params.source_folder}'
    base_task_path = f'{base_path}/{folder_path}'
    output_path = Path(f'{base_task_path}/single_feature_prediction')
    models_path = f'{base_task_path}/train_test_data'
    output_path.mkdir(exist_ok=True)
    learning_rate = param_dict['learning_rate']
    num_epochs = 500
    batch_size = param_dict['batch_size']
    layers_sized = param_dict.get("layers_sized")
    if layers_sized is None:
        layers_sized = [406, 4096, 2048, 1024, 512, 64, 1]
    label_name = "mean_performance"

    print('start')

    if task == 'digits':
        models_path = f"{base_task_path}/lightgbm_feature_selection/{selected_params.feature_selection_folder}/masked_data_models/"
        data_path = "2023-11-27-18-30-52_masked_data_200_features.pkl"
        with open(f'{models_path}/{data_path}', 'rb') as fp:
            data = joblib.load(fp)
        train_data = data['selected_test_data']
        test_data = data['selected_train_data']

    else:
        train_path = selected_params.train_data_path
        test_path = selected_params.test_data_path
        with open(f"{models_path}/{train_path}", 'rb') as fp:
            train_data = joblib.load(fp)
        with open(f"{models_path}/{test_path}", 'rb') as fp:
            test_data = joblib.load(fp)

    train_data = [(d[job_num - 1].reshape(-1), l) for d, l in train_data]
    test_data = [(d[job_num - 1].reshape(-1), l) for d, l in test_data]

    regression_lnn_learning(
        layers_sized=layers_sized,
        train_data=train_data,
        test_data=test_data,
        epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        test_every=1,
        output_path=output_path,
        task=task,
        save_model=False,
        sufix=f'feature_{job_num}',
    )
    print('end')
