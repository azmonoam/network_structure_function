import argparse
import multiprocessing
import os

import pandas as pd
from joblib import Parallel, delayed, dump

from new_organism import Organism
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from utils.main_utils import get_organism_from_pkl

num_cores = multiprocessing.cpu_count()


def _convert_org(
        file_name: str,
):
    if '/' in file_name:
        path = file_name
        file_name = file_name.split('/')[-1]
    else:
        path = f"{origin_full_path}/{file_name}"
    try:
        original_org = get_organism_from_pkl(
            path=path,
        )
    except:
        print(f"failed to open {file_name}")
        return
    max_allowed_connections = None
    if 'max_allowed_connections' in original_org.__dict__.keys():
        max_allowed_connections = original_org.max_allowed_connections
    new_org = Organism(
        dimensions=original_org.dimensions,
        num_layers=original_org.num_layers,
        allowed_weights_values=original_org.allowed_weights_values,
        allowed_bias_values=original_org.allowed_bias_values,
        connection_cost=original_org.connection_cost,
        predictions=original_org.predictions,
        performance=original_org.performance,
        fitness=original_org.fitness,
        generation=original_org.generation,
        network=original_org.network,
        biases=original_org.biases,
        use_bias=original_org.use_bias,
        layer_neuron_idx_mapping=original_org.layer_neuron_idx_mapping,
        max_allowed_connections=max_allowed_connections,
        communities_input_symmetry=communities_input_symmetry,
    )
    structural_features_calculator = CalcStructuralFeatures(
        organism=new_org,
    )
    new_org.structural_features.modularity = structural_features_calculator.get_modularity()
    new_org.structural_features.connectivity = structural_features_calculator.get_connectivity()
    new_org.structural_features.entropy = structural_features_calculator.get_entropy(
        num_connections=new_org.structural_features.connectivity.num_connections,
    )
    with open(f'{destination_full_path}/{file_name}', 'wb+') as fp:
        dump(new_org, fp)


def convert_all_orgs_in_folder(
        paths_from_csv: bool = False
) -> Organism:
    if paths_from_csv:
        files = pd.read_csv(origin_folder)['model_path'].to_list()
    else:
        files = os.listdir(origin_full_path)
    return Parallel(n_jobs=num_cores)(
            delayed
            (_convert_org)(file_name)
            for file_name in files
            if file_name not in os.listdir(destination_full_path)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_folder')
    parser.add_argument('--destination_path')
    args = parser.parse_args()
    origin_folder = args.origin_folder
    destination_path = args.destination_path
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    origin_full_path = f"{base_path}/{origin_folder}"
    destination_full_path = f"{base_path}/teach_archs/{destination_path}"
    communities_input_symmetry = 2
    convert_all_orgs_in_folder(
        paths_from_csv=True,
    )
