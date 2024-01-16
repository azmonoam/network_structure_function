import argparse
import os

import joblib
import pandas as pd

from parameters.selected_exp_names import selected_exp_names

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='xor')
parser.add_argument('--job_num', default=1)


args = parser.parse_args()
task = args.task
job_num = int(args.job_num)

# results_path = "/Volumes/noamaz/modularity/"
base_path = '/home/labs/schneidmann/noamaz/modularity'
size = 'random'
params = selected_exp_names[task][size]
dims = params.dims
res_folder = params.source_folder
exp_folder = params.feature_selection_folder
num_features = params.num_selected_features
model_folds = [
    f"feature_correlation/{num_features}_features/corr_clusters",
    f"feature_globality/{num_features}_features/masked_data_models"
]
data_folds = [
    f"feature_correlation/{num_features}_features/corr_clusters",
    f"feature_globality/{num_features}_features/teach_archs_regression_feature_selection_results_1kep"
]
all_features_selected  = []
for model in os.listdir(f"{base_path}/{res_folder}/{model_folds[job_num-1]}"):
    with open(
            f"{base_path}/{res_folder}/{model_folds[job_num-1]}/{model}", 'rb+') as fp:
        data_model = joblib.load(fp)
    selected_feature_names = set(data_model['selected_feature_names'])
    if selected_feature_names in all_features_selected:
        continue
    else:
        all_features_selected.append(selected_feature_names)
        model_name =  model.split('.pkl')[0]