import os

import joblib
import pandas as pd
from parameters.selected_exp_names import selected_exp_names

save_connectivity = False

max_num_features = 84
task_name = 'xor'
base_path = '/Volumes/noamaz/modularity'

selected_params = selected_exp_names[task_name]['random']
source_folder = f'/{task_name}/{selected_params.source_folder}'
num_features = selected_params.num_selected_features
exp_name = selected_params.feature_selection_folder
pkl_source_path = f"{source_folder}/lightgbm_feature_selection/{exp_name}/masked_data_models"

for model_name in os.listdir(f"{base_path}/{pkl_source_path}"):
    if model_name[0] == '.':
        continue
    num_featues_in_pkl = int(model_name.split("masked_data_")[1].split('_')[0])
    if num_featues_in_pkl == num_features:
        pkl_name = model_name
    elif num_featues_in_pkl == max_num_features:
        full_pkl_name = model_name

target_folder = f"{source_folder}/train_test_data"
target_pkl_name = pkl_name.replace('masked_data', 'all_train_test_masked_data')

with open(f"{base_path}/{pkl_source_path}/{pkl_name}", 'rb') as fp:
    masked_data = joblib.load(fp)

all_data = masked_data['selected_train_data'] + masked_data['selected_test_data']

with open(f"{base_path}/{target_folder}/{task_name}_{target_pkl_name}", 'wb+') as fp:
    joblib.dump(all_data, fp)
print(f"saved {num_features} data to: {base_path}/{target_folder}/{task_name}_{target_pkl_name}")

if save_connectivity:
    target_connevtivity_pkl_name = full_pkl_name.split('masked_data')[0] + 'all_train_test_connectivity_data'
    with open(f"{base_path}/{pkl_source_path}/{full_pkl_name}", 'rb+') as fp:
        full_data = joblib.load(fp)
    connectivity_ind = full_data['selected_feature_names'].index('connectivity_ratio')
    connectivities = pd.DataFrame(
        [
            data[connectivity_ind]
            for data, _ in full_data['selected_train_data'] + full_data['selected_test_data']
        ]
    ).astype(float)

    with open(f"{base_path}/{target_folder}/{task_name}_{target_connevtivity_pkl_name}.pkl", 'wb+') as fp:
        joblib.dump(connectivities, fp)
    print(
        f"saved {num_features} connectivity data to: {base_path}/{target_folder}/{task_name}_{target_connevtivity_pkl_name}")

print('a')