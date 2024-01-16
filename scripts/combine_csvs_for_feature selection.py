import os
import pandas as pd
from parameters.selected_exp_names import selected_exp_names


feature_selection = pd.DataFrame()
used_features = pd.DataFrame()
task_name = 'digits'
base_path = '/Volumes/noamaz/modularity'

selected_params = selected_exp_names[task_name]['random']
source_folder = f'/{task_name}/{selected_params.source_folder}'
exp_name = selected_params.feature_selection_folder
csvs_path = f"{base_path}/{source_folder}/lightgbm_feature_selection/{exp_name}/"
for csv_name in os.listdir(csvs_path):
    if ".csv" not in csv_name:
        continue
    res = pd.read_csv(f"{csvs_path}/{csv_name}", index_col=0)
    if 'feature_selection' in csv_name:
        feature_selection = pd.concat([res, feature_selection], ignore_index=True)
    elif 'used_features' in csv_name:
        used_features = pd.concat([res, used_features], ignore_index=True)
feature_selection['num_features'] = feature_selection['num_features'].astype(int)
feature_selection = feature_selection.sort_values('num_features')
feature_selection.to_csv(f'{csvs_path}/2023-11-28-11-32-00_all_feature_selection.csv')
used_features.to_csv(f'{csvs_path}/2023-11-28-11-32-00_all_used_features.csv')
