import argparse
import random
from datetime import datetime as dt
from itertools import permutations

import joblib

from logical_gates import LogicalGates
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.xor.xor_by_dim import XoraByDim
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from teach_arch_multiple_times import TeachArchMultiTime
from utils.set_up_population_utils import get_organism_by_connectivity_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    parser.add_argument('--task', default='xor')
    parser.add_argument('--parallel', default='0')
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--num_layers', default='4')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task_name = args.task
    parallel = bool(int(args.parallel))
    n_threads = int(args.n_threads)
    num_layers = int(args.num_layers)

    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}_{job_num}'

    num_exp_per_arch = 300
    in_dim = 6
    out_dim = 2
    if task_name == 'xor':
        possible_dims = [
            # [6, 6, 2],
            [6, 5, 3],
            # [6, 4, 4],
            # [5, 5, 4],
        ]
    elif task_name == 'digits':
        in_dim = 64
        out_dim = 10
        possible_dims = [
            [6, 6],
            [8, 4],
            [7, 5],
            [9, 3],
            [10, 2],
        ]
    all_possible_middle_dims = []
    for possible_mid_dim in possible_dims:
        all_possible_middle_dims += list(permutations(possible_mid_dim))
    dims = [in_dim]
    middle = random.choice(list(set(all_possible_middle_dims)))
    dims += middle
    dims.append(out_dim)
    if task_name == 'xor':
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            num_epochs=5000,
        )
    elif task_name == 'retina':
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            num_epochs=5000,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
    elif task_name == 'digits':
        task_params = DigitsByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            num_epochs=2000,
        )
    else:
        raise ValueError
    task_params.task_version_name = f'{task_name}_{num_layers}_layers'
    # task_params.base_path = '/Volumes/noamaz/modularity'
    output_path = f"{task_params.output_folder}/{exp_name}_teach.csv"
    connectivity_ratio = round(random.uniform(task_params.min_connectivity, 1.0), 2)

    organism = get_organism_by_connectivity_ratio(
        task_params=task_params,
        connectivity_ratio=connectivity_ratio,
    )
    structural_features_calculator = CalcStructuralFeatures(
        organism=organism,
    )
    organism = structural_features_calculator.calc_structural_features()
    organism.normed_structural_features = structural_features_calculator.calc_normed_structural_features(
        parameters=task_params,
    )
    with open(f'{task_params.pkls_folder}/{exp_name}.pkl', 'wb+') as fp:
        joblib.dump(organism, fp)

    teach_arch = TeachArchMultiTime(
        output_path=output_path,
        exp_name=exp_name,
        model_cls=task_params.model_cls,
        learning_rate=task_params.learning_rate,
        num_epochs=task_params.num_epochs,
        num_exp_per_arch=num_exp_per_arch,
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
