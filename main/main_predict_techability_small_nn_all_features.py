import argparse
import os
from pathlib import Path

import joblib

from jobs_params import (
    predict_techability_retina_xor_after_feature_selection,
    predict_techability_xor_after_feature_selection,
    predict_techability_digits_after_feature_selection,
)
from networks_teachability_regression.regression_nn_learn import regression_lnn_learning
from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    #base_path = "/Volumes/noamaz/modularity/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=1)
    parser.add_argument('--task', default='retina')
    args = parser.parse_args()
    job_num = int(args.job_num)
    task = args.task
    if job_num != 0:
        task = [
            'retina_xor',
            'xor',
            'digits'
        ][job_num-1]

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

    exp_path = f"{base_task_path}/small_nn_reg"
    output_path = Path(f'{exp_path}')
    output_path.mkdir(exist_ok=True)
    models_path = f'{base_task_path}/train_test_data'
    learning_rate = param_dict['learning_rate']
    num_epochs = param_dict['num_epochs']
    batch_size = param_dict['batch_size']
    layers_sized = param_dict.get("layers_sized")
    if layers_sized is None:
        layers_sized = [406, 512, 1]
    label_name = "mean_performance"

    print('start')
    train_path = selected_params.train_data_path
    test_path = selected_params.test_data_path
    with open(f"{models_path}/{train_path}", 'rb') as fp:
        train_data = joblib.load(fp)
    with open(f"{models_path}/{test_path}", 'rb') as fp:
        test_data = joblib.load(fp)
    num_features = str(test_data[0][0].shape[0])

    sufix = f'_small_3_layer_nn_'

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
