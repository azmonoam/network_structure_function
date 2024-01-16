import argparse
import os

import joblib
import pandas as pd

from parameters.selected_exp_names import selected_exp_names

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='xor')
args = parser.parse_args()
task = args.task

# results_path = "/Volumes/noamaz/modularity/"
base_path = '/home/labs/schneidmann/noamaz/modularity'
size = 'random'
params = selected_exp_names[task][size]
dims = params.dims
res_folder = params.source_folder
exp_folder = params.feature_selection_folder
num_features = params.num_selected_features

nn_data_path = f"{base_path}/{task}/{res_folder}/lightgbm_feature_selection/{exp_folder}/masked_data_models"

for nn_data_file in os.listdir(nn_data_path):
    if '._' in nn_data_file:
        continue
    if f"data_{num_features}_features.pkl" in nn_data_file:
        break

with open(
        f"{nn_data_path}/{nn_data_file}", 'rb+') as fp:
    data_model = joblib.load(fp)

data = data_model['selected_test_data'] + data_model['selected_train_data']
selected_feature_names = data_model['selected_feature_names']
res = {}
for i, feature_name in enumerate(selected_feature_names):
    res[feature_name] = [sample[i].item() for sample, _ in data]
res['label'] = [label.item() / 1000 for _, label in data]

pd.DataFrame.from_dict(res).to_csv(
    f"{base_path}/{task}/{res_folder}/lightgbm_feature_selection/{exp_folder}/{num_features}_feature_values.csv",
)
