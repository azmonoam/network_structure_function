import argparse
import os
import subprocess
from datetime import datetime as dt
from typing import List

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from parameters.selected_exp_names import selected_exp_names


def abs_percentage_error_by_batch(A, F):
    abs_errors = [
        abs((a - f) / a)
        for a, f in zip(A, F)
    ]
    jumps = list(range(0, len(abs_errors), 512)) + [len(abs_errors)]
    means = [
        np.mean(abs_errors[jumps[i]:jumps[i + 1]])
        for i in range(len(jumps) - 1)
    ]
    return np.mean(means), np.var(abs_errors)


def get_model_pred_and_mape(
        nn_data_file: str,
        nn_data_path: str,
        nn_models_path: str,
        all_models_files: List[str],
        num_features: int,
):
    temp = pd.DataFrame()
    exp_name = nn_data_file.split('.pkl')[0]
    try:
        with open(f"{nn_data_path}/{nn_data_file}", 'rb+') as fp:
            data = joblib.load(fp)
    except:
        print(f"couldnt open {nn_data_path}/{nn_data_file}")
        return
    test_data = data.get('selected_test_data')
    train_data = data.get('selected_train_data')
    if num_features is not None and test_data[0][0].shape[0] != num_features:
        return
    ex_files = [nn_file for nn_file in all_models_files if exp_name in nn_file]
    if len(ex_files) == 0:
        if num_features is not None:
            exp_name = nn_data_file.split(f'{num_features}_features')[0]
            ex_files = [nn_file for nn_file in all_models_files if exp_name in nn_file]
        else:
            num_features = len(data.get('selected_feature_names'))
            ex_files = [
                nn_file for nn_file in all_models_files
                if f"_{num_features}_features" in nn_file
            ]
    if len(ex_files) == 0:
        df_data = {
            'exp_name': exp_name,
            'num_features': num_features,
            'train_mape': None,
            'train_mape_var': None,
            'test_mape': None,
            'test_mape_var': None,
        }
        df = pd.DataFrame(df_data, index=[0], )
    for nn_file in ex_files:
        if '._' in nn_file:
            continue
        if f"_best_model_cpu.pkl" in nn_file:
            try:
                with open(f"{nn_models_path}/{nn_file}", 'rb+') as fp:
                    model = joblib.load(fp)
            except:
                print(f"couldnt open {nn_models_path}/{nn_file}")
                continue
            test_loader = DataLoader(
                dataset=test_data,
                batch_size=512,
                shuffle=False,
            )
            res_test = pd.DataFrame()
            test_pred = []
            test_label_no_increse = []
            for test_input, test_label in test_loader:
                test_outputs = model(test_input)
                test_pred += (test_outputs.reshape(-1).detach() / 1000).tolist()
                test_label_no_increse += (test_label / 1000).tolist()
            res_test['test_pred'] = test_pred
            res_test['test_label'] = test_label_no_increse

            train_loader = DataLoader(
                dataset=train_data,
                batch_size=512,
                shuffle=False,
            )
            res_train = pd.DataFrame()
            train_pred = []
            train_label_no_increse = []
            for train_input, train_label in train_loader:
                train_outputs = model(train_input)
                train_pred += (train_outputs.reshape(-1).detach() / 1000).tolist()
                train_label_no_increse += (train_label / 1000).tolist()
            res_train['train_pred'] = train_pred
            res_train['train_label'] = train_label_no_increse

            out_csv_path = f"{nn_models_path}/{exp_name}_prediction_results"
            res_test.to_csv(
                f"{out_csv_path}_test.csv"
            )
            res_train.to_csv(
                f"{out_csv_path}_train.csv"
            )
            test_mape, test_var = abs_percentage_error_by_batch(res_test['test_label'], res_test['test_pred'])
            train_mape, train_var = abs_percentage_error_by_batch(res_train['train_label'], res_train['train_pred'])
            df_data = {
                'exp_name': exp_name,
                'num_features': num_features,
                'train_mape': train_mape,
                'train_mape_var': train_var,
                'test_mape': test_mape,
                'test_mape_var': test_var,
            }
            df = pd.DataFrame(df_data, index=[0], )
        elif '_output_best.csv' in nn_file:
            temp = pd.read_csv(f"{nn_models_path}/{nn_file}", index_col=False)

    return pd.concat([df, temp], axis=1)


def parallel_wrapper(
        n_cores: int,
        nn_data_path: str,
        nn_models_path: str,
        num_features: int,
):
    all_models_files = [f for f in os.listdir(nn_models_path) if '_prediction_results' not in f]
    return joblib.Parallel(
        n_jobs=n_cores,
        timeout=9999,
    )(
        joblib.delayed
        (get_model_pred_and_mape)(
            nn_data_file=nn_data_file,
            nn_data_path=nn_data_path,
            all_models_files=all_models_files,
            nn_models_path=nn_models_path,
            num_features=num_features,
        )
        for nn_data_file in os.listdir(path=nn_data_path)
        if '._' not in nn_data_file
    )


time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='retina_xor')
parser.add_argument('--out_folder', default='teach_archs_regression_feature_selection_results')
parser.add_argument('--job_num', default=1)
parser.add_argument('--type_ind', default=0)
parser.add_argument('--num_cores', default=1)
parser.add_argument('--by_glob_level', default=1)

args = parser.parse_args()
job_num = int(args.job_num)
type_ind = int(args.type_ind)
by_glob_level = bool(int(args.by_glob_level))
try:
    n_cores = int(subprocess.run('nproc', capture_output=True).stdout)
except FileNotFoundError:
    n_cores = int(args.num_cores)

if job_num == 0:
    task = args.task
else:
    task = [
        'xor',
        'digits',
        'retina_xor',
    ][job_num - 1]

out_folder = args.out_folder

base_path = '/home/labs/schneidmann/noamaz/modularity'
# base_path = "/Volumes/noamaz/modularity/"

size = 'random'
params = selected_exp_names[task][size]
dims = params.dims
res_folder = params.source_folder
num_features = params.num_selected_features
if by_glob_level:
    type_folder = "lightgbm_feature_selection/by_globality"
    exp_folder = [
        f
        for f in os.listdir(f"{base_path}/{task}/{res_folder}/{type_folder}")
        if f not in ['archive', 'with_inv']
    ][type_ind]
    name_add = f'_{exp_folder.split("_")[-1]}'
    num_features = None
else:
    type_folder = [
        'feature_correlation',
        'feature_globality',
        'random_feature_selection',
        'lightgbm_feature_selection',
    ][type_ind]
    exp_folder = f"{num_features}_features"
    name_add = ''
    if type_ind == 2:
        out_folder = 'teach_archs_regression_random_feature_selection_results'
    elif type_ind == 3:
        exp_folder = params.feature_selection_folder
        out_folder = 'teach_archs_regression_feature_selection_results_with_preds_by_mae'
        num_features = None
    if task == 'xor':
        out_folder = 'teach_archs_regression_feature_selection_results_with_preds'
if task == 'digits':
    out_folder = 'teach_archs_regression_feature_selection_results_1kep'
    if type_ind == 3:
        out_folder = 'teach_archs_regression_feature_selection_results_1kep_with_preds'

nn_models_path = f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/{out_folder}"
nn_data_path = f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/masked_data_models"

all_res = pd.DataFrame()
res_list = parallel_wrapper(
    n_cores=n_cores,
    nn_models_path=nn_models_path,
    nn_data_path=nn_data_path,
    num_features=num_features,
)
for temp in res_list:
    print(temp)
    all_res = pd.concat([all_res, temp], ignore_index=True)

all_res.to_csv(
    f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/{time_str}_all_exp_mape_prediction_results{name_add}.csv",
)
