import random
from datetime import datetime as dt
from typing import List, Tuple

import joblib
import networkx as nx
import torch
import numpy as np
import pandas as pd


from new_organism import Organism
from parameters.retina_parameters import general_allowed_bias_values, \
    general_allowed_weights_values
from teach_arch_multiple_times import TeachArchMultiTime
from find_feature_dist_utils.tasks_params import TaskParameters
from stractural_features_models.structural_features import StructuralFeatures

class FindFeaturesSpectrum:
    def __init__(
            self,
            num_features: int,
            train_path: str,
            test_path: str,
            used_features_csv_name: str,
            results_folder: str,
            base_path: str,
            task_params: TaskParameters,
            target_label: str = "mean performance",
    ):
        self.target_label = target_label
        self.task_params = task_params
        self.results_path = f"{base_path}/teach_archs/{task_params.task_name}"
        self.results_folder = results_folder
        self.time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.exp_name_template = '{time}_required_pref_{min_labels_range}_{max_labels_range}_density_{density}'
        self.num_features = num_features
        self.all_data = self._get_all_archs_data(
            train_path=f"{self.results_path}/{train_path}",
            test_path=f"{self.results_path}/{test_path}"
        )
        selected_features_df = pd.read_csv(f"{self.results_path}/{used_features_csv_name}").drop("Unnamed: 0",
                                                                                                 axis=1)
        selected_features = selected_features_df[selected_features_df.sum(axis=1) == num_features]
        self.selected_feature_names = selected_features[selected_features == 1].dropna(axis=1).columns
        self.mask_tensor = torch.tensor(selected_features.iloc[0]).to(torch.bool)
        structural_features_data, self.adj_vecs = self._get_archs_and_structural_features_data_lists()
        self.structural_features_df = self._get_structural_features_df(
            structural_features_data_list=structural_features_data,
        )

    @staticmethod
    def _get_all_archs_data(
            train_path: str,
            test_path: str,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        with open(f"{train_path}", 'rb') as fp:
            train_data = joblib.load(fp)
        with open(f"{test_path}", 'rb') as fp:
            test_data = joblib.load(fp)
        all_data = train_data + test_data
        random.shuffle(all_data)
        return all_data

    def _get_selected_features_name(self) -> List[str]:
        selected_feature_names_with_label = self.selected_feature_names.to_list()
        selected_feature_names_with_label.append(self.target_label)
        return selected_feature_names_with_label

    def _get_archs_and_structural_features_data_lists(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        adj_vecs = []
        structural_features_data = []
        end_of_adj_vec_ind = sum(self.task_params.start_dimensions) ** 2
        for inputs, label_val in self.all_data:
            adj_vec, structural_features = inputs.split(end_of_adj_vec_ind)
            structural_features_data.append(
                torch.cat((torch.masked_select(structural_features, self.mask_tensor), label_val.reshape(-1))).tolist())
            adj_vecs.append(adj_vec)
        return structural_features_data, adj_vecs

    def _get_structural_features_df(
            self,
            structural_features_data_list: List[torch.Tensor],
    ) -> pd.DataFrame:
        selected_feature_names_with_label = self._get_selected_features_name()
        return pd.DataFrame(structural_features_data_list, columns=selected_feature_names_with_label)

    def _get_max_allowed_connections(
            self,
            connectivity_ratio: float,
    ):
        fully_connected_num_connection = sum(
            self.task_params.start_dimensions[i] * self.task_params.start_dimensions[i + 1]
            for i in range(self.task_params.num_layers)
        )
        return int(np.floor(fully_connected_num_connection * connectivity_ratio))

    def _build_organism(
            self,
            connectivity_ratio: float,
    ) -> Organism:
        max_allowed_connections = self._get_max_allowed_connections(
            connectivity_ratio=connectivity_ratio
        )
        organism = Organism(
            dimensions=self.task_params.start_dimensions,
            num_layers=self.task_params.num_layers,
            allowed_weights_values=general_allowed_weights_values,
            allowed_bias_values=general_allowed_bias_values,
            communities_input_symmetry=self.task_params.communities_input_symmetry,
            max_allowed_connections=max_allowed_connections,
            network=nx.DiGraph(),
            structural_features=StructuralFeatures(),
        )
        organism.build_organism_by_connectivity(
            max_allowed_connections=max_allowed_connections,
        )
        return organism

    def tech_arch_many_times(
            self,
            exp_name: str,
            organism: Organism,
            output_path: str,
            num_exp_per_arch: int,
    ):
        teach_arch = TeachArchMultiTime(
            exp_name=exp_name,
            output_path=output_path,
            model_cls=self.task_params.model_cls,
            learning_rate=self.task_params.learning_rate,
            num_epochs=self.task_params.num_epochs,
            num_exp_per_arch=num_exp_per_arch,
            task=self.task_params.task,
            activate=self.task_params.activate,
        )

        teach_arch.teach_arch_many_times_parallel(
            organism=organism,
        )

    def save_aggregated_results(
            self,
            exp_name: str,
            output_path: str,
            labels_range: Tuple[float, float],
            density: float,
    ):
        results = pd.read_csv(output_path).drop("Unnamed: 0", axis=1)
        final_epoch_res = results[results['iterations'] == results['iterations'].max()]
        first_analysis = {
            'exp_name': exp_name,
            'median_performance': final_epoch_res['performance'].median(),
            'mean_performance': final_epoch_res['performance'].mean(),
            'performance_std': final_epoch_res['performance'].std(),
            'max_performance': final_epoch_res['performance'].max(),
            'required_performance_min': labels_range[0] / 1000,
            'required_performance_max': labels_range[1] / 1000,
            'is_within_required_performance_range':
                (labels_range[0] / 1000 <= final_epoch_res['performance'].mean() <= labels_range[1] / 1000),
            'density': density,
        }
        pd.DataFrame.from_dict(first_analysis, orient='index').to_csv(
            f"{self.results_path}/{self.results_folder}/results/{exp_name}_res_analysis.csv",
        )
