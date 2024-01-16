from typing import List, Tuple, Optional

import numpy as np

from ergm_model.data_halper import get_list_of_functions_from_features_names
from ergm_model.ergm import Ergm
from ergm_model.ergm_utils import get_allowed_connections_based_on_ff_network_dims
from ergm_model.methods import Methods
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance


class ErgmClassic(Ergm):
    def __init__(
            self,
            features_names: List[str],
            find_feature_dist: Optional[FindFeaturesDistByPerformance] = None,
            target_mean_features: Optional[List[float]] = None,
            dimensions: Optional[List[int]] = None,
            n_nodes: Optional[int] = None,
            example_graphs_path: Optional[str] = None,
            example_graphs: Optional[np.ndarray] = None,
            initial_coeffs_values: Optional[List[float]] = None,
            restrict_allowed_connections: Optional[bool] = False,
            errors: Optional[np.ndarray] = None,
            n_threads: int = 8,
            num_isolated_neurones_feature: Optional[int] = None,
            distance_early_stopping_criteria_num_sigmas: float = 0.1,
            num_graphs_to_sample_for_early_stopping: int = 200,
            mse_early_stopping_criteria_factor: float = 0.005,
            early_stopping_type: str = 'distance',
            num_stds: int = 1,
            seperation: int = 5
    ):
        super().__init__(
            features_names, find_feature_dist, target_mean_features, n_nodes, restrict_allowed_connections, errors,
            n_threads, num_isolated_neurones_feature, distance_early_stopping_criteria_num_sigmas,
            num_graphs_to_sample_for_early_stopping, mse_early_stopping_criteria_factor, early_stopping_type, num_stds,
            seperation
        )
        self.dimensions = dimensions
        if self.dimensions is not None:
            self.n_nodes = sum(dimensions)
            self.allowed_connections = get_allowed_connections_based_on_ff_network_dims(
                dimensions=self.dimensions,
            )
        self.methods = Methods(
            dimensions=dimensions,
        )
        self.features = get_list_of_functions_from_features_names(
            method_class=self.methods,
            features_names_list=features_names,
        )
        self._init_coeffs(
            initial_coeffs_values=initial_coeffs_values,
            example_graphs_path=example_graphs_path,
            example_graphs=example_graphs,
        )
        print(f'initial coefs:  {self.coefs}')
        print(f'target mean features:  {self.target_mean_features}')

    def sample_new_graph(
            self,
            coefs: np.ndarray,
            warmap: int = 0,
    ) -> Tuple[float, np.ndarray, List[float]]:
        current_graph = np.zeros((self.n_nodes, self.n_nodes))
        current_stats = self.calc_stats(
            graph=current_graph,
        )
        sum_delta_stats = [0 for f in self.features]
        if self.restrict_allowed_connections:
            for itter in range(self.seperation):
                if itter < warmap:
                    sum_delta_stats = [0 for f in self.features]
                for i, j in self.allowed_connections:
                    current_graph, current_stats, sum_delta_stats = self._update_ergm(
                        coefs=coefs,
                        current_graph=current_graph,
                        i=i,
                        j=j,
                        current_stats=current_stats,
                        sum_delta_stats=sum_delta_stats,
                    )
        else:
            for itter in range(self.seperation):
                if itter < warmap:
                    sum_delta_stats = [0 for f in self.features]
                for i in range(self.n_nodes):
                    for j in range(i + 1, self.n_nodes):
                        current_graph, current_stats, sum_delta_stats = self._update_ergm(
                            coefs=coefs,
                            current_graph=current_graph,
                            i=i,
                            j=j,
                            current_stats=current_stats,
                            sum_delta_stats=sum_delta_stats,
                        )
        return sum_delta_stats, current_graph, current_stats
