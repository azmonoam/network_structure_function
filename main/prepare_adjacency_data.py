import multiprocessing
import random
from typing import Tuple, Optional, List

import networkx as nx
import torch
import numpy as np

from joblib import Parallel, delayed

from new_organism import Organism
from parameters.general_paramters import RANDOM_SEED


class PrepareAdjacencyData:

    def __init__(
            self,
            base_path: str,
            n_threads: Optional[int] = None,
    ):
        self.num_cores = n_threads
        if self.num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        self.base_path = base_path

    @staticmethod
    def get_network_aj_matrix(
            organism: Organism,
            weight: Optional[str] = 'weight',
            datatype: Optional[np.dtype] = float,
    ):
        return nx.to_numpy_array(organism.network, weight=weight, dtype=datatype)

    def create_test_and_train_data(
            self,
            train_percent: float = 0.7,
            features_list: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        self.exp_and_label = self._gather_data()
        data = self._prepare_data_from_csv_wrapper(
            features_list=features_list,
        )
        if len(data) == 0:
            raise 'data empty'
        fixed_data = [
            sample for sample in data if sample is not None
        ]
        if len(fixed_data) != len(data):
            print(f'metadata vec length: {len(data) - len(fixed_data)} samples had a problem in length')
        random.seed(RANDOM_SEED)
        random.shuffle(fixed_data)
        partial_len = int(np.floor(len(fixed_data) * train_percent))
        train_data = fixed_data[:partial_len]
        test_data = fixed_data[partial_len:]
        return train_data, test_data

    def _prepare_data_from_csv_wrapper(
            self,
            features_list: Optional[List[str]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return Parallel(n_jobs=self.num_cores)(
            delayed
            (self._prepare_data_from_csv)(exp_data, features_list)
            for exp_data in self.exp_and_label
        )

    def _gather_data(self):
        pass

    def _prepare_data_from_csv(
            self,
            exp_data,
            features_list: Optional[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
