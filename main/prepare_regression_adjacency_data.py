from datetime import datetime
from typing import Tuple, Optional, List

import joblib
import pandas as pd
import torch

from prepare_adjacency_data import PrepareAdjacencyData


class PrepareRegressionAdjacencyData(PrepareAdjacencyData):

    def __init__(
            self,
            base_path: str,
            results_csv_name: Optional[str],
            results_model_path: Optional[str],
            label_name: Optional[str],
            increase_label_scale: Optional[bool],
            consider_meta_data: Optional[bool],
            consider_adj_mat: bool,
            normalize_features: bool = False,
            structural_features_vec_length: Optional[int] = None,
            n_threads: Optional[int] = None,
    ):
        super().__init__(base_path, n_threads)
        self.results_csv_name = results_csv_name
        self.label_name = label_name
        self.results_model_path = f'{base_path}/{results_model_path}'
        self.increase_label_scale = increase_label_scale
        self.consider_meta_data = consider_meta_data
        self.consider_adj_mat = consider_adj_mat
        self.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.structural_features_vec_length = structural_features_vec_length
        self.normalize_features = normalize_features

    def _prepare_data_from_csv(
            self,
            exp_data: Tuple[str, float],
            features_list: Optional[List[str]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        exp_name, label = exp_data
        label = torch.tensor(
            label,
        ).float()
        try:
            with open(f"{self.results_model_path}/{str(exp_name)}.pkl", 'rb') as fp:
                organism = joblib.load(fp)
        except:
            print(f'Organism in path: f"{self.results_model_path}/{str(exp_name)}.pkl" could not be open')
            return
        if self.increase_label_scale:
            label = label * 1000
        if self.consider_meta_data:
            if not self.normalize_features:
                meta_data_ = organism.structural_features.get_class_values(
                    features_list=features_list,
                )
            else:
                meta_data_ = organism.normed_structural_features.get_class_values(
                    features_list=features_list,
                )
            meta_data = torch.Tensor(
                meta_data_
            ).float()
            input_vec = meta_data
            if self.structural_features_vec_length is not None:
                if input_vec.shape[0] != self.structural_features_vec_length:
                    with open(f"{self.time_str}_problematic_pkls.txt", "a") as file:
                        file.write(f"name: {self.results_model_path}/{str(exp_name)}.pkl, len: {input_vec.shape[0]}\n")
                    return None
        if self.consider_adj_mat:
            adj_mat = self.get_network_aj_matrix(
                organism=organism,
                weight=None,
            )
            adj_vec = torch.reshape(
                torch.from_numpy(
                    adj_mat
                ),
                (-1,)
            ).float()
            input_vec = adj_vec
        if self.consider_meta_data and self.consider_adj_mat:
            input_vec = torch.cat((adj_vec, meta_data), 0)
        return (
            input_vec,
            label,
        )

    def _gather_data(self):
        if self.results_csv_name:
            first_analysis_df = pd.read_csv(f"{self.base_path}/{self.results_csv_name}").drop("Unnamed: 0", axis=1,  errors='ignore')
            exp_and_label = [
                tuple(x)
                for x in first_analysis_df[
                    [
                        'exp_name',
                        self.label_name
                    ]
                ].values
            ]
        else:
            exp_and_label = []
        return exp_and_label
