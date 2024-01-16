import argparse
import os
import subprocess
from datetime import datetime as dt

import joblib
import pandas as pd

from parameters.selected_exp_names import selected_exp_names


def get_model_dropouts(
        nn_data_file: str,
        nn_data_path: str,
):
    try:
        with open(f"{nn_data_path}/{nn_data_file}", 'rb+') as fp:
            data = joblib.load(fp)
    except:
        print(f"couldnt open {nn_data_path}/{nn_data_file}")
        return
    all_data = [d for d, l in data.get('selected_test_data') + data.get('selected_train_data')]
    num_features = len(data['selected_feature_names'])
    data_df = pd.DataFrame(all_data).astype("float")
    res = pd.DataFrame()
    ls =[data_df.shape[0]]
    ns = [num_features]
    rs = [-1]
    parts = [1]
    for i in range(2, 7):
        data_df_ = data_df.round(decimals=i).drop_duplicates()
        ns.append(num_features)
        rs.append(i)
        ls.append(data_df_.shape[0])
        parts.append(round(ls[-1]/ ls[0], 3))
    res[f'num_rows'] = ls
    res['num_features'] = ns
    res['rounding_ratio'] = rs
    res['size_ratio'] =parts
    return res


def parallel_wrapper(
        n_cores: int,
        nn_data_path: str,

):
    return joblib.Parallel(
        n_jobs=n_cores,
        timeout=9999,
    )(
        joblib.delayed
        (get_model_dropouts)(
            nn_data_file=nn_data_file,
            nn_data_path=nn_data_path,
        )
        for nn_data_file in os.listdir(path=nn_data_path)
        if '._' not in nn_data_file
    )


time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='retina_xor')
parser.add_argument('--out_folder', default='teach_archs_regression_feature_selection_results')
parser.add_argument('--job_num', default=1)
parser.add_argument('--num_cores', default=1)

args = parser.parse_args()
job_num = int(args.job_num)
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
#base_path = "/Volumes/noamaz/modularity/"

size = 'random'
params = selected_exp_names[task][size]
dims = params.dims
res_folder = params.source_folder
num_features = params.num_selected_features
exp_folder = params.feature_selection_folder
type_folder = 'lightgbm_feature_selection'
if task == 'digits':
    out_folder = 'teach_archs_regression_feature_selection_results_1kep'

nn_models_path = f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/{out_folder}"
nn_data_path = f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/masked_data_models"

all_res = pd.DataFrame()
res_list = parallel_wrapper(
    n_cores=n_cores,
    nn_data_path=nn_data_path,
)
for temp in res_list:
    print(temp)
    all_res = pd.concat([all_res, temp], ignore_index=True)

all_res.to_csv(
    f"{base_path}/{task}/{res_folder}/{type_folder}/{exp_folder}/{time_str}_num_features_uniq_decay.csv",
)
