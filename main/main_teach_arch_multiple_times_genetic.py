import argparse
import os

import joblib

from logical_gates import LogicalGates
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.xor.xor_by_dim import XoraByDim
from teach_arch_multiple_times import TeachArchMultiTime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--task', default='retina')
    parser.add_argument('--parallel', default='0')
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--num_layers', default='3')
    parser.add_argument('--folder_name', default='4_features')
    parser.add_argument('--res_folder_name', default='teach_archs_results')
    parser.add_argument('--models_folder_name', default='teach_archs_models')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task_name = args.task
    parallel = bool(int(args.parallel))
    n_threads = int(args.n_threads)
    num_layers = int(args.num_layers)
    folder_name = args.folder_name
    res_folder_name = args.res_folder_name
    models_folder_name = args.models_folder_name

    num_epochs = 5000

    num_exp_per_arch = 300
    if task_name == 'xor':
        dims = [6, 6, 4, 4, 2]
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
        task_params.task_version_name = f'{task_name}_{num_layers}_layers'
        task_base_path = f'{task_params.base_path}/{task_params.task_global_name}/{task_params.task_version_name}'

    elif task_name == 'retina_xor':
        dims = [6, 5, 2, 2]
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
        task_params.task_version_name = f'retina_{num_layers}_layers'
        task_base_path = f'{task_params.base_path}/{task_params.task_base_folder_name}/{task_params.task_version_name}'

    elif task_name == 'digits':
        dims = [64, 6, 6, 10]
        task_params = DigitsByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            num_epochs=2000,
        )
        num_epochs = 2000
        task_params.task_version_name = f'{task_name}_{num_layers}_layers'
        task_base_path = f'{task_params.base_path}/{task_params.task_global_name}/{task_params.task_version_name}'
    else:
        raise ValueError
    # task_params.base_path = '/Volumes/noamaz/modularity'
    results_path = f'{task_base_path}/requiered_features_genetic_models/{folder_name}/good_archs/per_dim_results'
    models_folder = f'{results_path}/{models_folder_name}'
    results_folder = f"{results_path}/{res_folder_name}"
    model_name = [f for f in sorted(os.listdir(models_folder)) if '._' not in f][job_num - 1].split('.pkl')[0]
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
