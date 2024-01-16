import argparse
import asyncio

import pandas as pd

from logical_gates import LogicalGates
from teach_arch_multiple_times import TeachArchMultiTime
from utils.main_utils import get_organism_from_pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)
    args = parser.parse_args()
    job_num = int(args.job_num)
    input_dim = 6
    num_exp_per_arch = 300
    learning_rate = 0.1
    rule = LogicalGates.AND
    num_epochs = 2000
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    successful_results_path = f'{base_path}/successful_results_4'
    files = pd.read_csv(f"{base_path}/successful_results_4_pkl_names.csv")['files']
    pkl_path = files[job_num]
    exp_name = pkl_path.split('.pkl')[0]
    output_path = f"{base_path}/teach_arch_4_2/{exp_name}_teach.csv"
    input_path = f"{successful_results_path}/{pkl_path}"
    teach_arch = TeachArchMultiTime(
        input_dim=input_dim,
        output_path=output_path,
        exp_name=exp_name,
        rule=rule,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_exp_per_arch=num_exp_per_arch
    )
    try:
        organism = get_organism_from_pkl(
            path=input_path,
        )
    except Exception as e:
        print(f'Organism in path: {input_path} could not be open')
        raise
    asyncio.run(
        teach_arch.teach_arch_many_times_parallel(
            organism=organism,
        )
    )
