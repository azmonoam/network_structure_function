import argparse

import joblib
import pandas as pd

from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim
from teach_arch_multiple_times import TeachArchMultiTime
from parameters.xor.xor_by_dim import XoraByDim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--task', default='retina')
    parser.add_argument('--parallel', default='1')
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--num_layers', default='3')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task_name = args.task
    parallel = bool(int(args.parallel))
    n_threads = int(args.n_threads)
    num_layers = int(args.num_layers)
    resave_org = True

    if task_name == 'retina':
        dims = [6, 5, 2, 2]
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
        original_csv_name = '2023-08-20-12-33-37_first_analsis_general_no_duplicates.csv'
        original_location = 'retina/retina_3_layers_5_2/teach_archs_models'
    elif task_name == 'xor':
        dims = [6, 6, 5, 3, 2]
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
        resave_org = True
        original_location = 'xor/xor_f/teach_archs_models'
        original_csv_name = '2023-08-08-15-32-28_first_analsis_general_no_duplicates.csv'
    else:
        raise ValueError()

   # task_params.base_path = '/Volumes/noamaz/modularity'
    base_path = task_params.base_path
    main_exp_path = task_params.teach_arch_base_path

    original_csv = pd.read_csv(f"{main_exp_path}/{original_csv_name}")
    exp_name = original_csv['exp_name'].iloc[job_num - 1]

    output_path = f"{task_params.output_folder}/{exp_name}_teach.csv"
    if resave_org:
        with open(f'{base_path}/{original_location}/{exp_name}.pkl', 'rb') as fp:
            organism = joblib.load(fp)

        with open(f'{task_params.pkls_folder}/{exp_name}.pkl', 'wb+') as fp:
            joblib.dump(organism, fp)
    else:
        with open(f'{task_params.pkls_folder}/{exp_name}.pkl', 'rb') as fp:
            organism = joblib.load(fp)

    teach_arch = TeachArchMultiTime(
        output_path=output_path,
        exp_name=exp_name,
        model_cls=task_params.model_cls,
        learning_rate=task_params.learning_rate,
        num_epochs=task_params.num_epochs,
        num_exp_per_arch=300,
        task=task_params.task,
        activate=task_params.activate,
        flatten=task_params.flatten,
        n_threads=n_threads,
        optimizer=task_params.optimizer,
        test_every=200,
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
