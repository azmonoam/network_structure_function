import argparse
import os

from joblib import Parallel, delayed, load, dump

from stractural_features_models.calc_structural_features import CalcStructuralFeatures


def _add_structural_features(
        csv_name: str,
        models_folder: str,
        models_out_folder: str,
):
    exp_name = csv_name.split('_teach.csv')[0]
    if exp_name in os.listdir(models_out_folder):
        return
    try:
        with open(f"{models_folder}/{exp_name}.pkl", 'rb') as fp:
            organism = load(fp)
    except:
        print(f'Organism in path: f"{models_folder}/{exp_name}.pkl" could not be open')
        return
    structural_features_calculator = CalcStructuralFeatures(
        organism=organism,
    )
    organism.structural_features.modularity = structural_features_calculator.get_modularity()
    organism.structural_features.connectivity = structural_features_calculator.get_connectivity()
    organism.structural_features.entropy = structural_features_calculator.get_entropy(
        num_connections=organism.structural_features.connectivity.num_connections,
    )
    with open(f"{models_out_folder}/{exp_name}.pkl", 'wb+') as fp:
        dump(organism, fp)


def _add_structural_features_wrapper(
        num_cores: int,
        results_folder: str,
        models_folder: str,
        models_out_folder: str,
):
    print(f"-- using {results_folder} as folder name --")
    return Parallel(
        n_jobs=num_cores,
        timeout=9999,
    )(
        delayed
        (_add_structural_features)(
            csv_name=csv_name,
            models_folder=models_folder,
            models_out_folder=models_out_folder,
        )
        for csv_name in os.listdir(path=results_folder)
        if os.path.exists(f"{results_folder}/{csv_name}")
    )


if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity/teach_archs'
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default='1')

    args = parser.parse_args()
    base_folder = 'digits'
    res_folder = 'digits_teach_archs_results'
    models_folder = f"{base_path}/{base_folder}/digits_teach_archs_models"
    models_out_folder = f"{base_path}/{base_folder}/digits_teach_archs_new_models"
    num_cores = int(args.n_threads)
    results_folder = f"{base_path}/{base_folder}/{res_folder}"
    print(f"-- using {num_cores} cores --")

    _add_structural_features_wrapper(
        num_cores=num_cores,
        results_folder=results_folder,
        models_folder=models_folder,
        models_out_folder=models_out_folder,
    )
