import argparse
import os

import joblib
import pandas as pd
from torch.utils.data import DataLoader

from parameters.selected_exp_names import selected_exp_names

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='retina_xor')
parser.add_argument('--out_folder', default='teach_archs_regression_feature_selection_results')
parser.add_argument('--max_n_features', default='0')
parser.add_argument('--job_num', default=0)

args = parser.parse_args()
job_num = int(args.job_num)
if job_num == 0:
    task = args.task
else:
    task = [
        'xor',
        'digits',
        'retina_xor',
    ][job_num - 1]

out_folder = args.out_folder
max_n_features = bool(int(args.max_n_features))

base_path = '/home/labs/schneidmann/noamaz/modularity'
# base_path = "/Volumes/noamaz/modularity/"
if task == 'digits':
    out_folder = 'teach_archs_regression_feature_selection_results_1kep'
size = 'random'
params = selected_exp_names[task][size]
dims = params.dims
res_folder = params.source_folder
exp_folder = params.feature_selection_folder
max_num_features = {
    'xor': 5,
    'retina_xor': 6,
    'digits': 20,
}
if max_n_features:
    num_features = max_num_features[task]
else:
    num_features = params.num_selected_features

nn_models_path = f"{base_path}/{task}/{res_folder}/lightgbm_feature_selection/{exp_folder}/{out_folder}"
nn_data_path = f"{base_path}/{task}/{res_folder}/lightgbm_feature_selection/{exp_folder}/masked_data_models"

for nn_file in os.listdir(nn_models_path):
    if '._' in nn_file:
        continue
    if f"_{num_features}_features_best_model_cpu.pkl" in nn_file:
        break
with open(f"{nn_models_path}/{nn_file}", 'rb+') as fp:
    model = joblib.load(fp)
print(f"{nn_models_path}/{nn_file}")

for nn_data_file in os.listdir(nn_data_path):
    if '._' in nn_data_file:
        continue
    if f"data_{num_features}_features.pkl" in nn_data_file:
        break

with open(f"{nn_data_path}/{nn_data_file}", 'rb+') as fp:
    test_data = joblib.load(fp).get('selected_test_data')
print(f"{nn_data_path}/{nn_data_file}")

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
out_csv_path = f"{base_path}/{task}/{res_folder}/lightgbm_feature_selection/{exp_folder}/{num_features}_prediction_results.csv"
print(out_csv_path)
res.to_csv(
    out_csv_path
)
