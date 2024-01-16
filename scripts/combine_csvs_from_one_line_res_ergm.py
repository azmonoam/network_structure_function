import os
from datetime import datetime as dt

import pandas as pd


# For adding "res analsis" files of one column of few lines with mean performance results - all exp per arch combined
base_path = '/home/labs/schneidmann/noamaz/modularity'
base_path = '/Volumes/noamaz/modularity'
task_name = 'retina'

folder_name = '5_features'
results_path = f"{base_path}/teach_archs/{task_name}"
results_path = f'{results_path}/{task_name}_teach_archs_requiered_features_ergm/{folder_name}/2023-07-22-14-40-54'
results_folder ='results'
csvs_path = f"{results_path}/{results_folder}"
time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
columns = [
    'exp_name',
    'median_performance',
    'mean_performance',
    'performance_std',
    'max_performance',
    'required_performance_min',
    'required_performance_max',
    'is_within_required_performance_range',
]
all_res = pd.DataFrame(columns=columns)
for csv_name in os.listdir(csvs_path):
    if "res_analysis.csv" not in csv_name:
        continue
    res = pd.read_csv(f"{csvs_path}/{csv_name}", index_col=0).T
    all_res = pd.concat([all_res, res], ignore_index=True)
path = f"{results_path}/{time_str}_all_results_on_ergm_combined"
#path = f"{results_path}//{time_str}_all_results_combined"
all_res.to_csv(f'{path}.csv')
all_res_no_duplicates = all_res.drop_duplicates(subset=[
    'modularity',
    'num_connections',
    'entropy',
    'normed_entropy',
    'density',
])
all_res_no_duplicates.to_csv(f'{path}_no_duplicates.csv')
print(path)