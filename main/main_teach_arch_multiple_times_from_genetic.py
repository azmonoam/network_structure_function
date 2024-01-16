import argparse
import os

import joblib
import pandas as pd

from teach_arch_multiple_times import TeachArchMultiTime
from utils.tasks_params import RetinaParameters

base_path = '/home/labs/schneidmann/noamaz/modularity'
# base_path = '/Volumes/noamaz/modularity'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--n_threads', default=1)
    parser.add_argument('--folder_name')
    parser.add_argument('--target')
    args = parser.parse_args()
    job_num = int(args.job_num)
    n_threads = int(args.n_threads)
    folder_name = args.folder_name
    target = args.target

    task = 'retina'
    task_params = RetinaParameters
    num_exp_per_arch = 300
    if target == 'min_performance':
        required_performance_min, required_performance_max = (0.753002197265625, 0.8163546142578125)
    else:
        required_performance_min = 0.9363038330078125
        required_performance_max = 0.9522058715820313
    results_path = f'{base_path}/teach_archs/{task}/{task}_teach_archs_requiered_features_genetic/{folder_name}'
    models_folder = f'{results_path}/models'
    results_folder = f"{results_path}/results"
    model_name = sorted(os.listdir(models_folder))[job_num].split('.pkl')[0]
    if '2023-07-26' not in model_name or f"{model_name}_res_analysis.csv" in sorted(os.listdir(results_folder)):
        print(f'{model_name} already exist')
    else:
        with open(f"{models_folder}/{model_name}.pkl", 'rb') as fp:
            organism = joblib.load(fp)
        output_path = f"{results_path}/results/{model_name}_teach.csv"
        teach_arch = TeachArchMultiTime(
            exp_name=model_name,
            output_path=output_path,
            model_cls=task_params.model_cls,
            learning_rate=task_params.learning_rate,
            num_epochs=task_params.num_epochs,
            num_exp_per_arch=num_exp_per_arch,
            task=task_params.task,
            activate=task_params.activate,
            n_threads=n_threads,
        )

        teach_arch.teach_arch_many_times_parallel(
            organism=organism,
        )
        results = pd.read_csv(output_path).drop("Unnamed: 0", axis=1)
        first_analysis = {
            'exp_name': model_name,
            'median_performance': results['performance'].median(),
            'mean_performance': results['performance'].mean(),
            'performance_std': results['performance'].std(),
            'max_performance': results['performance'].max(),
            'required_performance_min': required_performance_min,
            'required_performance_max': required_performance_max,
            'is_within_required_performance_range':
                (required_performance_min <= results['performance'].mean() <= required_performance_max),
            'density': organism.structural_features.connectivity.connectivity_ratio,
            'modularity': results['modularity'].iloc[0],
            'num_connections': results['num_connections'].iloc[0],
            'entropy': results['entropy'].iloc[0],
            'normed_entropy': results['normed_entropy'].iloc[0],
        }
        pd.DataFrame.from_dict(first_analysis, orient='index').to_csv(
            f"{results_folder}/{model_name}_res_analysis.csv",
        )
        print(f'done {model_name}')
