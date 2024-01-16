import os
from datetime import datetime
from typing import Tuple, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load

from parameters.retina_parameters import retina_structural_features_full_name_vec


class GetNetworkTeachabilityData:
    def __init__(
            self,
            num_cores: int,
            folder: str,
            models_folder: str,
            res_folder: str,
            extended_results: bool,
            add_motifs: bool,
            new_models: bool,
            add_normlized: bool,
    ):
        self.WANTED_CONNECTIVITY_KEYS = [
            'total_connectivity_ratio_between_layers',
            'max_connectivity_between_layers_per_layer',
            'layer_connectivity_rank',
            'num_paths_to_output_per_input_neuron',
            'num_involved_neurons_in_paths_per_input_neuron',
        ]
        self.models_folder = models_folder
        self.num_cores = num_cores
        self.folder = folder
        self.extended_results = extended_results
        self.add_motifs = add_motifs
        self.new_models = new_models
        self.add_normlized = add_normlized
        self.full_name_vec = retina_structural_features_full_name_vec[3:]
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if extended_results:
            self.csv_name = f'{time_str}_all_extended_results_from_{res_folder.replace("/", "_")}'
        elif add_motifs:
            self.csv_name = f'{time_str}_all_results_from_{res_folder.replace("/", "_")}_with_motifs'
        else:
            self.csv_name = f'{time_str}_all_results_from_{res_folder.replace("/", "_")}'
        self._get_subset_columns_to_drop_dop()

    def _check_if_name_in_wanted_name_catagories(
            self,
            name: str,
    ) -> bool:
        if name.rsplit('_', 1)[0] in self.WANTED_CONNECTIVITY_KEYS:
            return True
        return False

    def _get_subset_columns_to_drop_dop(self):
        self.subset = [
            'modularity',
            'num_connections',
            'entropy',
            'normed_entropy',
            'connectivity_ratio',
            'num_neurons',
        ]
        if self.new_models:
            self.subset += [
                'neurons_in_layer_0',
                'neurons_in_layer_1',
                'neurons_in_layer_2',
                'neurons_in_layer_3',
                'max_possible_connections',
                'motifs_count_0',
                'motifs_count_1',
                'motifs_count_2',
            ]
        elif self.add_motifs:
            self.subset += [
                'motifs_count_0',
                'motifs_count_1',
                'motifs_count_2',
            ]

    def _get_motif_data(
            self,
            organism,
    ) -> dict:
        return {
            f'motifs_count_{i}': motif
            for i, motif in enumerate(organism.structural_features.motifs.motifs_count)
        }

    def _get_new_models_data(
            self,
            organism,
    ) -> dict:
        new_models_data_dict = self._get_motif_data(organism)
        for i, dim in enumerate(organism.structural_features.structure.dimensions):
            new_models_data_dict[f'neurons_in_layer_{i}'] = dim
        new_models_data_dict['num_layers'] = organism.structural_features.structure.num_layers
        new_models_data_dict['num_neurons'] = organism.structural_features.structure.num_neurons
        new_models_data_dict[
            'max_possible_connections'] = organism.structural_features.connectivity.max_possible_connections
        return new_models_data_dict

    def _add_normed_data(
            self,
            organism,
    ) -> dict:
        new_models_data_dict = {}
        for i, motif in enumerate(organism.normed_structural_features.motifs.motifs_count):
            new_models_data_dict[f'normalized_motifs_count_{i}'] = motif
        new_models_data_dict['normalized_entropy'] = organism.normed_structural_features.entropy.entropy
        new_models_data_dict[
            'normalized_normed_entropy'] = organism.normed_structural_features.entropy.normed_entropy
        return new_models_data_dict

    def _build_tabels_rows(
            self,
            csv_name: str,
            epoch: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        try:
            results = pd.read_csv(f"{self.folder}/{csv_name}").drop("Unnamed: 0", axis=1, errors='ignore')
        except:
            print(f"{self.folder}/{csv_name}")
            return
        if results.shape[0] <= 1:
            return
        exp_name = results['exp_name'].iloc[0]
        with open(f"{self.models_folder}/{exp_name}.pkl", 'rb') as fp:
            organism = load(fp)
        model_data = self._get_model_basic_data(
            organism=organism,
            exp_name=exp_name,
        )
        model_data = self._add_additional_data(
            organism=organism,
            first_analysis=model_data,
        )
        if epoch is None:
            epoch = results['iterations'].max()
        final_epoch_res = results[results['iterations'] == epoch]
        if final_epoch_res.shape[0] > 0:
            first_analysis = self._get_ephoc_data_from_df(
                first_analysis=model_data,
                final_epoch_res=final_epoch_res,
            )
            return first_analysis
        return None

    def _get_extended_results(
            self,
            organism,
    ):
        connectivity_dict = {
            self.full_name_vec[ind]: value
            for ind, value in enumerate(organism.structural_features.connectivity.get_class_values())
            if self._check_if_name_in_wanted_name_catagories(self.full_name_vec[ind])
        }
        connectivity_dict['mean_num_paths_to_output_per_input_neuron'] = \
            np.mean(organism.structural_features.connectivity.num_paths_to_output_per_input_neuron)
        connectivity_dict['mean_num_involved_neurons_in_paths_per_input_neuron'] = \
            np.mean(organism.structural_features.connectivity.num_involved_neurons_in_paths_per_input_neuron)
        connectivity_dict['num_neuron_combinations_that_have_a_connecting_path'] = sum(
            1
            for distance in organism.structural_features.connectivity.distances_between_input_neuron
            if distance != -1
        )
        connectivity_dict['num_neuron_combinations_that_dont_have_a_connecting_path'] = sum(
            1
            for distance in organism.structural_features.connectivity.distances_between_input_neuron
            if distance == -1
        )
        connectivity_dict['ratio_of_num_neuron_combinations_that_have_dont_have_a_connecting_path'] = (
            connectivity_dict['num_neuron_combinations_that_have_a_connecting_path'] /
            connectivity_dict['num_neuron_combinations_that_dont_have_a_connecting_path']
            if connectivity_dict['num_neuron_combinations_that_dont_have_a_connecting_path'] != 0 else 10
        )
        connectivity_dict['mean_distances_between_input_neuron'] = np.mean(
            [
                distance
                for distance in organism.structural_features.connectivity.distances_between_input_neuron
                if distance != -1
            ]
        )
        return connectivity_dict

    def _get_list_of_files_to_run(self):
        csvs = []
        for csv_name in os.listdir(path=self.folder):
            if self.existing_exps is not None:
                exp_name = csv_name.split("_teach.csv")[0]
                if exp_name in self.existing_exps:
                    continue
            csvs.append(csv_name)
        return csvs

    def _prepare_data_from_csv_wrapper(
            self,
            epoch: Union[Optional[int], List[int]] = None,
    ) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
        print(f"-- using {self.folder} as folder name --")
        csvs = self._get_list_of_files_to_run()
        return Parallel(
            n_jobs=self.num_cores,
            timeout=9999,
        )(
            delayed
            (self._build_tabels_rows)(csv_name, epoch)
            for csv_name in csvs
        )

    def combine_all_data(
            self,
            existing_exps_analysis_full_path: Optional[str],
            epoch: Union[Optional[int], List[int]] = None,
    ):
        if existing_exps_analysis_full_path is not None:
            first_analysis_df = pd.read_csv(existing_exps_analysis_full_path)
            self.existing_exps = first_analysis_df['exp_name'].to_list()
        else:
            first_analysis_df = pd.DataFrame()
            self.existing_exps = None
        data_list = self._prepare_data_from_csv_wrapper(epoch)
        for first_analysis_dict in data_list:
            if first_analysis_dict is not None:
                first_analysis_df = pd.concat([first_analysis_df, pd.DataFrame(first_analysis_dict, index=[0], )],
                                              ignore_index=True)
        return first_analysis_df

    def _test_one_file(
            self,
            csv_name: str,
            epoch: List[int] = None,
    ):
        results = pd.read_csv(f"{self.folder}/{csv_name}").drop("Unnamed: 0", axis=1, errors='ignore')
        for e in epoch:
            if results[results['iterations'] == e].shape[0] == 0:
                raise ValueError(f'Could not find results for epoch {e}, please redefine the epochs')

    def _get_ephoc_data_from_df(
            self,
            first_analysis: Dict[str, Any],
            final_epoch_res: pd.DataFrame,
    ) -> Dict[str, Any]:
        threshold = 1.0
        num_successes = final_epoch_res[final_epoch_res['performance'] >= threshold].shape[0]
        first_analysis_performance_data = {
            'median_performance': final_epoch_res['performance'].median(),
            'mean_performance': final_epoch_res['performance'].mean(),
            'performance_std': final_epoch_res['performance'].std(),
            'max_performance': final_epoch_res['performance'].max(),
            f'num_successes_1.0': num_successes,
            f'success_percent_1.0': num_successes / final_epoch_res.shape[0]
        }
        return {**first_analysis, **first_analysis_performance_data}

    @staticmethod
    def _get_model_basic_data(
            exp_name: str,
            organism,
    ) -> Dict[str, Any]:
        modularity = organism.structural_features.modularity.modularity
        num_connections = organism.structural_features.connectivity.num_connections
        connectivity_ratio = organism.structural_features.connectivity.connectivity_ratio
        entropy = organism.structural_features.entropy.entropy
        normed_entropy = organism.structural_features.entropy.normed_entropy
        return {
            'exp_name': exp_name,
            'modularity': modularity,
            'num_connections': num_connections,
            'entropy': entropy,
            'normed_entropy': normed_entropy,
            'connectivity_ratio': connectivity_ratio,
        }

    def _add_additional_data(
            self,
            first_analysis: Dict[str, Any],
            organism,
    ):
        if self.new_models:
            new_models_data_dict = self._get_new_models_data(organism)
            first_analysis = {**first_analysis, **new_models_data_dict}
            if self.add_normlized:
                new_models_data_dict = self._add_normed_data(organism)
                first_analysis = {**first_analysis, **new_models_data_dict}
        elif self.add_motifs:
            motif_data_dict = self._get_motif_data(organism)
            first_analysis = {**first_analysis, **motif_data_dict}
        if self.extended_results:
            connectivity_dict = self._get_extended_results(organism)
            first_analysis = {**first_analysis, **connectivity_dict}
        return first_analysis
