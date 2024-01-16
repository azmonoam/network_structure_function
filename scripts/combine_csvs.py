import os
from datetime import datetime as dt

import pandas as pd

# For combining and adding "teach" files of multiple lines with mean performance results  - multiple exp per arch combined



task_name = 'retina'
required_performance_min = 0.9363038330078125
required_performance_max = 0.9522058715820313
base_path = '/home/labs/schneidmann/noamaz/modularity'
base_path = '/Volumes/noamaz/modularity'
num_features =5
results_path = f"{base_path}/teach_archs/{task_name}"
results_path = f'{results_path}/{task_name}_teach_archs_requiered_features_genetic/{num_features}_features'
results_folder ='results'
csvs_path = f"{results_path}/{results_folder}"
time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
columns = [
    'exp_name',
    'modularity',
    'num_connections',
    'entropy',
    'normed_entropy',
    'median_performance',
    'mean_performance',
    'performance_std',
    'max_performance',
    'required_performance_min',
    'required_performance_max',
    'is_within_required_performance_range',
    'density'
]
all_res = pd.DataFrame(columns=columns)
for csv_name in os.listdir(csvs_path):
    if "_teach.csv" not in csv_name:
        continue
    res = pd.read_csv(f"{csvs_path}/{csv_name}", index_col=0)
    first_analysis = {
        'exp_name': res['exp_name'].iloc[0],
        'modularity': res['modularity'].iloc[0],
        'num_connections': res['num_connections'].iloc[0],
        'entropy': res['entropy'].iloc[0],
        'normed_entropy': res['normed_entropy'].iloc[0],
        'median_performance': res['performance'].median(),
        'mean_performance': res['performance'].mean(),
        'performance_std': res['performance'].std(),
        'max_performance': res['performance'].max(),
        'required_performance_min': required_performance_min,
        'required_performance_max': required_performance_max,
        'is_within_required_performance_range':
            (required_performance_min <= res[
                'performance'].mean() <= required_performance_max),
        'density': res['connectivity_ratio'].iloc[0],
    }
    all_res = pd.concat([all_res,  pd.DataFrame(first_analysis, index=[0])], ignore_index=True)

all_res_no_duplicates = all_res.drop_duplicates(subset=[

    'modularity',
    'num_connections',
    'entropy',
    'normed_entropy',
    'density',
])
path = f"{results_path}/{results_folder}/{time_str}_all_results_combined_no_dop.csv"
all_res_no_duplicates.to_csv(path)
print(path)
