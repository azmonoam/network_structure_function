import random
from typing import List, Optional, Union, Tuple

import numpy as np
from joblib import load

from ergm_model.random_dim.ergm_random_dim import ErgmRandomdim
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance


class ErgmReferenceGraphRandomDim(ErgmRandomdim):
    def __init__(
            self,
            features_names: List[str],
            find_feature_dist: Optional[FindFeaturesDistByPerformance] = None,
            target_mean_features: Optional[List[float]] = None,
            n_nodes: Optional[int] = None,
            input_size: Optional[int] = None,
            output_size: Optional[int] = None,
            num_layers: Optional[int] = None,
            example_graphs_path: Optional[str] = None,
            example_graphs: Optional[List[List[Union[np.ndarray, Tuple[int, ...]]]]] = None,
            initial_coeffs_values: Optional[List[float]] = None,
            restrict_allowed_connections: Optional[bool] = False,
            errors: Optional[np.ndarray] = None,
            n_threads: int = 8,
            num_isolated_neurones_feature: Optional[int] = None,
    ):
        super().__init__(features_names, find_feature_dist, target_mean_features, n_nodes, input_size, output_size,
                         num_layers, example_graphs_path, example_graphs, initial_coeffs_values,
                         restrict_allowed_connections, errors, n_threads, num_isolated_neurones_feature)

    def _init_coeffs(
            self,
            initial_coeffs_values: Optional[List[float]] = None,
            example_graphs_path: Optional[str] = None,
            example_graphs: Optional[List[List[Union[np.ndarray, Tuple[int, ...]]]]] = None,
    ):
        self.coefs = initial_coeffs_values
        if self.coefs is None:
            self.coefs, self.example_graphs = self._get_initial_coeffs_values(
                example_graphs_path=example_graphs_path,
                example_graphs=example_graphs,
            )
        else:
            self.example_graphs = example_graphs
            if self.example_graphs is None:
                self.example_graphs = self._get_example_graphs(
                    example_graphs_path=example_graphs_path,
                )

    def _get_example_graphs(
            self,
            example_graphs_path: Optional[str],
    ) -> Optional[List[List[Union[np.ndarray, Tuple[int, ...]]]]]:
        if example_graphs_path is None:
            return None
        with open(example_graphs_path, 'rb') as fp:
            example_graphs = load(fp)
        return example_graphs

    def _get_initial_graph_and_dim(self):
        return random.choice(self.example_graphs)
