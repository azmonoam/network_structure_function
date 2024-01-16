import argparse
from pathlib import Path

import joblib

from jobs_params import (
    predict_techability_retina_xor_after_feature_selection,
    predict_techability_xor_after_feature_selection,
    predict_techability_digits_after_feature_selection,
)
from networks_teachability_regression.logistic_regression import logistic_regression_learning
from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    # base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    base_path = '/home/labs/schneidmann/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--task', default='retina')
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

    lambda_reg = [0.5, 1, 0.2, 0][job_num - 1]

    selected_params = selected_exp_names[task]['random']
    folder_path = f'{task}/{selected_params.source_folder}'
    base_task_path = f'{base_path}/{folder_path}'

    exp_path = f"{base_task_path}/logistic_regression_results"
    output_path = Path(f'{exp_path}')
    output_path.mkdir(exist_ok=True)
    models_path = f'{base_task_path}/train_test_data'
    learning_rate = param_dict['learning_rate']
    num_epochs = param_dict['num_epochs']
    batch_size = param_dict['batch_size']
    layers_size = [406, 1]
    label_name = "mean_performance"

    print('start')
    train_path = selected_params.train_data_path
    test_path = selected_params.test_data_path
    with open(f"{models_path}/{train_path}", 'rb') as fp:
        train_data = joblib.load(fp)
    with open(f"{models_path}/{test_path}", 'rb') as fp:
        test_data = joblib.load(fp)
    num_features = str(test_data[0][0].shape[0])
    sufix = f'log_reg_lamda_{lambda_reg}'
    print('start')
    logistic_regression_learning(
        layers_size=layers_size,
        train_data=train_data,
        test_data=test_data,
        lambda_reg=lambda_reg,
        epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        test_every=1,
        sufix=sufix,
        output_path=output_path,
        task=task,
    )
    print('end')
