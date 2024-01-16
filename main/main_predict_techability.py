import argparse
import os

from jobs_params import (
    main_predict_techability_xor,
    main_predict_techability_retina,
    main_predict_techability_params_retina_multi_arch_3_layers,
    main_predict_techability_params_retina_multi_arch_4_layers,
    main_predict_techability_digits,
    main_predict_techability_params_retina_xor,
    main_predict_techability_params_xor_new,
    main_predict_techability_params_digis_new
)
from networks_teachability_regression.regression_nn_learn import regression_lnn_learning, get_data

if __name__ == '__main__':
    # base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=1)
    parser.add_argument('--task', default='retina')
    parser.add_argument('--layers', default=3)
    parser.add_argument('--dyn_pkl_name', default=0)
    args = parser.parse_args()
    job_num = int(args.job_num)
    layers = int(args.layers)
    dyn_pkl_name = bool(int(args.dyn_pkl_name))
    task = args.task
    if task == 'retina':
        if layers == 3:
            param_dict = main_predict_techability_params_retina_multi_arch_3_layers
        elif layers == 4:
            param_dict = main_predict_techability_params_retina_multi_arch_4_layers
        else:
            param_dict = main_predict_techability_retina
    elif task == 'xor':
        param_dict = main_predict_techability_params_xor_new
    elif task == 'retina_xor':
        param_dict = main_predict_techability_params_retina_xor
    elif task == 'digits':
        param_dict = main_predict_techability_params_digis_new
    else:
        raise ValueError()
    folder_path = param_dict["folder_path"]
    data_folder = param_dict["data_folder"]
    output_path = f'{base_path}/{folder_path}/{param_dict["output_path"]}'
    if dyn_pkl_name:
        version = sorted(os.listdir(f'{base_path}/{folder_path}/{data_folder}'))[job_num - 1]
        num_ep = version.split('_')[-2]
        train_path = f'{base_path}/{folder_path}/{data_folder}/{version.replace("test", "train")}'
        test_path = f'{base_path}/{folder_path}/{data_folder}/{version.replace("train", "test")}'
        sufix = param_dict['sufix'] + f"_{num_ep}_ep"
        learning_rate = param_dict['learning_rate']
        num_epochs = param_dict['num_epochs']
        batch_size = param_dict['batch_size']
        layers_sized = param_dict.get("layers_sized")
    else:
        train_path = f'{base_path}/{folder_path}/{data_folder}/{param_dict[job_num]["train_path"]}'
        test_path = f'{base_path}/{folder_path}/{data_folder}/{param_dict[job_num]["test_path"]}'
        sufix = param_dict[job_num]['sufix']
        learning_rate = param_dict[job_num]['learning_rate']
        num_epochs = param_dict[job_num]['num_epochs']
        batch_size = param_dict[job_num]['batch_size']
        layers_sized = param_dict[job_num].get("layers_sized")
    results_csv_name = param_dict.get('results_csv_name')
    results_model_path = param_dict.get('results_model_path')

    if layers_sized is None:
        layers_sized = [406, 4096, 2048, 1024, 512, 64, 1]

    label_name = "mean_performance"
    print(output_path)
    print('start')
    train_data, test_data = get_data(
        results_csv_name=results_csv_name,
        results_model_path=results_model_path,
        label_name=label_name,
        base_path=base_path,
        train_path=train_path,
        test_path=test_path,
    )
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
