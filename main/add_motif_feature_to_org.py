import argparse
import os

from joblib import Parallel, delayed, load, dump

from stractural_features_models.calc_structural_features import CalcStructuralFeatures


def _add_structural_features(
        pkl_name: str,
        models_folder: str,
        models_out_folder: str,
):
    if pkl_name in os.listdir(models_out_folder):
        return
    try:
        with open(f"{models_folder}/{pkl_name}", 'rb') as fp:
            organism = load(fp)
    except:
        print(f'Organism in path: f"{models_folder}/{pkl_name}" could not be open')
        return
    structural_features_calculator = CalcStructuralFeatures(
        organism=organism,
    )
    organism.structural_features.motifs = structural_features_calculator.get_motifs()
    with open(f"{models_out_folder}/{pkl_name}", 'wb+') as fp:
        dump(organism, fp)


def _add_structural_features_wrapper(
        num_cores: int,
        models_folder: str,
        models_out_folder: str,
):
    print(f"-- using {models_folder} as folder name --")
    return Parallel(
        n_jobs=num_cores,
        timeout=9999,
    )(
        delayed
        (_add_structural_features)(
            pkl_name=pkl_name,
            models_folder=models_folder,
            models_out_folder=models_out_folder,
        )
        for pkl_name in os.listdir(path=models_folder)
        if os.path.exists(f"{models_folder}/{pkl_name}")
    )


if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity/teach_archs'
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default='1')

    args = parser.parse_args()
    base_folder = 'digits'
    models_folder = f"{base_path}/{base_folder}/digits_teach_archs_new_models"
    models_out_folder = f"{base_path}/{base_folder}/digits_teach_archs_new_models_with_motifs"
    num_cores = int(args.n_threads)
    print(f"-- using {num_cores} cores --")

    _add_structural_features_wrapper(
        num_cores=num_cores,
        models_folder=models_folder,
        models_out_folder=models_out_folder,
    )
