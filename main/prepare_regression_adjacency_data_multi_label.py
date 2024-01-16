import os
import random
from datetime import datetime
from typing import Tuple, Optional, List, Dict

import joblib
import torch

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from parameters.general_paramters import RANDOM_SEED
from prepare_adjacency_data import PrepareAdjacencyData


class PrepareRegressionAdjacencyDataMultiLabel(PrepareAdjacencyData):

    def __init__(
            self,
            base_path: str,
            results_csvs_dir: Optional[str],
            results_model_path: Optional[str],
            label_name: Optional[str],
            increase_label_scale: Optional[bool],
            consider_meta_data: Optional[bool],
            consider_adj_mat: bool,
            normalize_features: bool = False,
            n_threads: Optional[int] = None,
            eph_to_skip: Optional[List[str]] = None
    ):
        super().__init__(base_path, n_threads)
        self.results_csvs_dir = results_csvs_dir
        self.label_name = label_name
        self.results_model_path = f'{base_path}/{results_model_path}'
        self.increase_label_scale = increase_label_scale
        self.consider_meta_data = consider_meta_data
        self.consider_adj_mat = consider_adj_mat
        self.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.normalize_features = normalize_features
        self.eph_to_skip = eph_to_skip

    def create_test_and_train_data(
            self,
            train_percent: float = 0.7,
            features_list: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        random.seed(RANDOM_SEED)
        self.exp_and_label = self._gather_data()
        data = self._prepare_data_from_csv_wrapper(
            features_list=features_list,
        )
        partial_len = int(np.floor(len(data) * train_percent))
        train_inds = random.sample(range(0, len(data)), partial_len)
        train_test_all_ep_data = {}
        for epochs_num in list(data[0].keys()):
            train_test_all_ep_data[epochs_num] = {
                'train': [],
                'test': [],
            }
            for i in range(len(data)):
                if i in train_inds:
                    train_test_all_ep_data[epochs_num]['train'].append(data[i][epochs_num])
                else:
                    train_test_all_ep_data[epochs_num]['test'].append(data[i][epochs_num])
        return train_test_all_ep_data

    def _prepare_multi_eph_data_from_csv(
            self,
            exp_data: pd.DataFrame,
            features_list: Optional[List[str]],
    ) -> Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        all_eph_data = {}
        for c in exp_data.index:
            if c == 'exp_name':
                exp_name = exp_data['exp_name']
                try:
                    with open(f"{self.results_model_path}/{str(exp_name)}.pkl", 'rb') as fp:
                        organism = joblib.load(fp)
                except:
                    print(f'Organism in path: f"{self.results_model_path}/{str(exp_name)}.pkl" could not be open')
                    return

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
            else:
                label = torch.tensor(
                    exp_data[c],
                ).float()
                if self.increase_label_scale:
                    label = label * 1000
                all_eph_data[c] = (
                    input_vec,
                    label,
                )
        return all_eph_data

    def _gather_data(self):
        exp_and_labels = None
        if not self.results_csvs_dir:
            return
        for results_csv_name in os.listdir(f"{self.base_path}/{self.results_csvs_dir}"):
            num_ep = results_csv_name.split("_")[-2]
            if self.eph_to_skip is not None and num_ep in self.eph_to_skip:
                continue
            first_analysis_df = pd.read_csv(f"{self.base_path}/{self.results_csvs_dir}/{results_csv_name}").drop(
                "Unnamed: 0", axis=1, errors='ignore')
            first_analysis_df.set_index('exp_name', drop=False, inplace=True)
            if exp_and_labels is None:
                exp_and_labels = first_analysis_df['exp_name']
            label_data = first_analysis_df[[self.label_name, ]]
            label_data = label_data.rename(columns={self.label_name: num_ep})
            exp_and_labels = pd.concat([exp_and_labels, label_data], axis=1)
        return exp_and_labels

    def _prepare_data_from_csv_wrapper(
            self,
            features_list: Optional[List[str]],
    ) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        return Parallel(n_jobs=self.num_cores)(
            delayed
            (self._prepare_multi_eph_data_from_csv)(self.exp_and_label.iloc[i], features_list)
            for i in range(self.exp_and_label.shape[0])
        )
