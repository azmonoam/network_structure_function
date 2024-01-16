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

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    #base_path = "/Volumes/noamaz/modularity/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=1)
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
    exp_path = param_dict["exp_path"]
    exp_full_path = f'{base_path}/{exp_path}/{param_dict["exp_folder"]}'
    output_path = Path(f'{exp_full_path}/{param_dict.get("output_path", "teach_archs_regression_feature_selection_results")}')
    output_path.mkdir(exist_ok=True)
    models_path = f'{exp_full_path}/{param_dict.get("model_path", "masked_data_models")}'
    learning_rate = param_dict['learning_rate']
    num_epochs = param_dict['num_epochs']
    batch_size = param_dict['batch_size']
    layers_sized = param_dict.get("layers_sized")
    if layers_sized is None:
        layers_sized = [406, 4096, 2048, 1024, 512, 64, 1]
    label_name = "mean_performance"

    print('start')
    model_name = [name for name in sorted(os.listdir(models_path)) if name[0] != '.'][job_num-1]

    with open(f"{models_path}/{model_name}", 'rb') as fp:
        masked_model = joblib.load(fp)
    train_data = masked_model.get('selected_train_data')
    test_data = masked_model.get('selected_test_data')
    num_features = masked_model.get('num_features')

    sufix = f'_meta_only_{num_features}_features{param_dict.get("sufix", "")}'

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
        save_preds=True
    )
    print('end')
