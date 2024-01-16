import argparse
import os

from joblib import Parallel, delayed, load, dump

from stractural_features_models.structural_features import *


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
    organism.structural_features = StructuralFeatures(
        modularity=ModularityFeatures(**organism.structural_features.modularity.__dict__),
        connectivity=ConnectivityFeatures(
            connectivity_ratio=organism.structural_features.connectivity.connectivity_ratio,
            num_connections=organism.structural_features.connectivity.num_connections,
            max_possible_connections=organism.structural_features.connectivity.max_possible_connections,
            connectivity_data_dict_per_layer=organism.structural_features.connectivity.__dict__,
            distances_between_input_neuron=organism.structural_features.connectivity.distances_between_input_neuron,
            num_paths_to_output_per_input_neuron=organism.structural_features.connectivity.num_paths_to_output_per_input_neuron,
            num_involved_neurons_in_paths_per_input_neuron=organism.structural_features.connectivity.num_involved_neurons_in_paths_per_input_neuron,
        ),
        entropy=EntropyFeatures(
            **organism.structural_features.entropy.__dict__,
        ),
        motifs=MotifsCount(
            **organism.structural_features.motifs.__dict__,
        ),
        structure=Structure(
            dimensions=organism.structural_features.structure.dimensions,
            num_layers=organism.structural_features.structure.num_layers,
        )
    )
    organism.normed_structural_features = StructuralFeatures(
        modularity=ModularityFeatures(**organism.normed_structural_features.modularity.__dict__),
        connectivity=ConnectivityFeatures(
            connectivity_ratio=organism.normed_structural_features.connectivity.connectivity_ratio,
            num_connections=organism.normed_structural_features.connectivity.num_connections,
            max_possible_connections=organism.normed_structural_features.connectivity.max_possible_connections,
            connectivity_data_dict_per_layer=organism.normed_structural_features.connectivity.__dict__,
            distances_between_input_neuron=organism.normed_structural_features.connectivity.distances_between_input_neuron,
            num_paths_to_output_per_input_neuron=organism.normed_structural_features.connectivity.num_paths_to_output_per_input_neuron,
            num_involved_neurons_in_paths_per_input_neuron=organism.normed_structural_features.connectivity.num_involved_neurons_in_paths_per_input_neuron,
        ),
        entropy=EntropyFeatures(
            **organism.normed_structural_features.entropy.__dict__,
        ),
        motifs=MotifsCount(
            **organism.normed_structural_features.motifs.__dict__,
        ),
        structure=Structure(
            dimensions=organism.normed_structural_features.structure.dimensions,
            num_layers=organism.normed_structural_features.structure.num_layers,
        )
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
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--base_folder', default='retina')


    args = parser.parse_args()
    base_folder = args.base_folder
    res_folder = 'teach_archs_results'
    models_folder = f"{base_path}/{base_folder}/teach_archs_models"
    models_out_folder = f"{base_path}/{base_folder}/teach_archs_models_new"
    num_cores = int(args.n_threads)
    results_folder = f"{base_path}/{base_folder}/{res_folder}"
    print(f"-- using {num_cores} cores --")

    _add_structural_features_wrapper(
        num_cores=num_cores,
        results_folder=results_folder,
        models_folder=models_folder,
        models_out_folder=models_out_folder,
    )
