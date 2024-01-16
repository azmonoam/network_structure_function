import argparse
import multiprocessing

from joblib import Parallel, delayed, dump, load

from new_organism import Organism
from parameters.retina_parameters import retina_structural_features_vec_length
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from typing import List

num_cores = multiprocessing.cpu_count()


def _update_org(
        file_name: str,
):
    if '/' in file_name:
        path = file_name
        file_name = file_name.split('/')[-1]
    else:
        path = f"{full_path}/{file_name}"
    try:
        with open(path, 'rb') as fp:
            original_org = load(fp)
    except:
        print(f"failed to open {file_name}")
        return
    structural_features_calculator = CalcStructuralFeatures(
        organism=original_org,
    )
    if len(original_org.structural_features.get_class_values()) == retina_structural_features_vec_length:
        return
    original_org.structural_features.connectivity = structural_features_calculator.get_connectivity()
    if len(original_org.structural_features.get_class_values()) != retina_structural_features_vec_length:
        print(f"length is still wrong: {len(original_org.structural_features.get_class_values())} - {file_name}")
        return
    with open(path, 'wb+') as fp:
        dump(original_org, fp)


def convert_all_orgs_in_folder(
        pathes: List[str]
) -> Organism:
    return Parallel(n_jobs=num_cores)(
        delayed
        (_update_org)(file_name)
        for file_name in pathes
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_folder')
    args = parser.parse_args()
    origin_folder = args.origin_folder
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    full_path = f"{base_path}/teach_archs/{origin_folder}"
    communities_input_symmetry = 3
    pathes = []
    with open(f"/home/labs/schneidmann/noamaz/network_modularity/2023-04-13-17-18-45_problematic_pkls.txt", 'r') as fp:
        problematic_pkls = fp
        for line in problematic_pkls:
            pathes.append(line.split(',')[0].split(' ')[1])
    convert_all_orgs_in_folder(pathes)
