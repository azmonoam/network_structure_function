import argparse
import os
from datetime import datetime as dt

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


time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='retina_xor')
parser.add_argument('--out_folder', default='teach_archs_regression_feature_selection_results')
parser.add_argument('--job_num', default=3)
parser.add_argument('--type_ind', default=1)

args = parser.parse_args()
job_num = int(args.job_num)
type_ind = int(args.type_ind)

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
base_path = "/Volumes/noamaz/modularity/"
if task == 'digits':
    out_folder = 'teach_archs_regression_feature_selection_results_1kep'
size = 'random'
params = selected_exp_names[task][size]
dims = params.dims
res_folder = params.source_folder
num_features = params.num_selected_features
type_folder = [
    'feature_correlation',
    'feature_globality',
    'random_feature_selection'
][type_ind]
if type_ind == 2:
    out_folder = 'teach_archs_regression_random_feature_selection_results'

exp_folder = f"{num_features}_features"

nn_models_path = f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/{out_folder}"
nn_data_path = f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/masked_data_models"

all_res = pd.DataFrame()
i = 0
all_models_files = os.listdir(nn_models_path)
while i < 10:
    for nn_data_file in os.listdir(nn_data_path):
        if '._' in nn_data_file:
            continue
        exp_name = nn_data_file.split('.pkl')[0]
        try:
            with open(f"{nn_data_path}/{nn_data_file}", 'rb+') as fp:
                test_data = joblib.load(fp)
        except:
            print(f"couldnt open {nn_data_path}/{nn_data_file}")
            i += 1
            continue
        test_data = test_data.get('selected_test_data')
        if test_data[0][0].shape[0] != num_features:
            continue
        temp = pd.DataFrame()
        mape, var = (None, None)
        ex_files = [nn_file for nn_file in all_models_files if exp_name in nn_file]
        if len(ex_files) == 0:
            exp_name = nn_data_file.split(f'_{num_features}_features')[0]
            ex_files = [nn_file for nn_file in all_models_files if exp_name in nn_file]
        for nn_file in ex_files:
            if '._' in nn_file:
                continue
            if f"_best_model_cpu.pkl" in nn_file:
                try:
                    with open(f"{nn_models_path}/{nn_file}", 'rb+') as fp:
                        model = joblib.load(fp)
                except:
                    print(f"couldnt open {nn_models_path}/{nn_file}")
                    i += 1
                    continue
                test_loader = DataLoader(
                    dataset=test_data,
                    batch_size=512,
                    shuffle=False,
                )
                res = pd.DataFrame()
                test_pred = []
                test_label_no_increse = []
                for test_input, test_label in test_loader:
                    test_outputs = model(test_input)
                    test_pred += (test_outputs.reshape(-1).detach() / 1000).tolist()
                    test_label_no_increse += (test_label / 1000).tolist()
                res['test_pred'] = test_pred
                res['test_label'] = test_label_no_increse
                out_csv_path = f"{nn_models_path}/{exp_name}_prediction_results.csv"
                res.to_csv(
                    out_csv_path
                )
                mape, var = abs_percentage_error_by_batch(res['test_label'], res['test_pred'])
            elif '_output_best.csv' in nn_file:
                temp = pd.read_csv(f"{nn_models_path}/{nn_file}", index_col=False)

        temp['exp_name'] = exp_name
        temp['mape'] = mape
        temp['mape_var'] = var
        all_res = pd.concat([all_res, temp], ignore_index=True)

all_res.to_csv(
    f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/{time_str}_all_exp_mape_prediction_results.csv",
)
