import argparse
import os

import joblib

from logical_gates import LogicalGates
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim
from teach_arch_multiple_times import TeachArchMultiTime

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--task', default='retina')
    parser.add_argument('--parallel', default='0')
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--exp_folder', default='per_dim_results')
    parser.add_argument('--random_init', default='0')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task_name = args.task
    parallel = bool(int(args.parallel))
    n_threads = int(args.n_threads)
    exp_folder = args.exp_folder
    random_init = bool(int(args.random_init))

    selected_params = selected_exp_names[task_name]['random']
    task_base_path = f'{base_path}/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    num_layers = selected_params.num_layers
    num_exp_per_arch = 300
    all_m = True
    if task_name == 'xor':
        dims = [6, 6, 5, 3, 2]
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
        num_epochs = 5000
    elif task_name == 'retina_xor':
        dims = [6, 3, 4, 2]
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
        num_epochs = 5000
    elif task_name == 'digits':
        dims = [64, 6, 6, 10]
        task_params = DigitsByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            num_epochs=2000,
        )
        num_epochs = 2000
        all_m = False
    else:
        raise ValueError
    results_path = f'{task_base_path}/ergm/{num_features}_features/{exp_folder}'
    if random_init:
        models_folder = f'{results_path}/teach_archs_models_random_init'
        results_folder = f"{results_path}/teach_archs_results_random_init"
    else:
        models_folder = f'{results_path}/teach_archs_models'
        results_folder = f"{results_path}/teach_archs_results"
    if all_m:
        model_name = sorted(os.listdir(models_folder))[job_num - 1].split('.pkl')[0]
    else:
        all_model_name = set([f.split('.pkl')[0] for f in sorted(os.listdir(models_folder))])
        all_res_name = set([f.split('_teach.csv')[0] for f in sorted(os.listdir(results_folder))])
        model_name = list(all_model_name-all_res_name)[job_num - 1]
    output_path = f"{results_folder}/{model_name}_teach.csv"

    with open(f'{models_folder}/{model_name}.pkl', 'rb') as fp:
        organism = joblib.load(fp)

    teach_arch = TeachArchMultiTime(
        output_path=output_path,
        exp_name=model_name,
        model_cls=task_params.model_cls,
        learning_rate=task_params.learning_rate,
        num_epochs=num_epochs,
        num_exp_per_arch=num_exp_per_arch,
        task=task_params.task,
        activate=task_params.activate,
        flatten=task_params.flatten,
        n_threads=n_threads,
        optimizer=task_params.optimizer,
    )

    if parallel:
        print('--- running parallel ---')
        teach_arch.teach_arch_many_times_parallel(
            organism=organism,
            return_all_the_way=True,
        )
    else:
        teach_arch.teach_arch_many_times(
            organism=organism,
            return_all_the_way=True,
        )
    print('DONE')
