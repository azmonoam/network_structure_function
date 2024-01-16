from typing import List, Optional

import numpy as np

from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from main.ergm_model.ergm_from_reference_graph import ErgmReferenceGraph


class ErgmAnyPointFromRef(ErgmReferenceGraph):
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
    ):
        super().__init__(features_names, find_feature_dist, target_mean_features, dimensions, n_nodes,
                         example_graphs_path, example_graphs, initial_coeffs_values, restrict_allowed_connections,
                         errors, n_threads, num_isolated_neurones_feature)
        self.rounding_funcs = [
            self._round_to_int,
            self._round_to_int,
            self._round_to_int,
            self._round_to_int,
            self._round_to_int,
            self._round_to_int,
        ]

    def calc_error_percentage_of_std(
            self,
            mean_obs_stats,
            early_stopping_criteria,
    ):
        ste = [
            abs(mean_obs_stat - func(mean_obs_stat)) / std
            for mean_obs_stat, func, std
            in zip(mean_obs_stats, self.rounding_funcs, self.errors)
        ]
        is_within_stopping_criteria = [
            mse < limit
            for mse, limit in zip(ste, early_stopping_criteria)
        ]
        print(f'error: {ste}')
        print(f'sum error: {sum(ste)}')
        print(f'is good enough ratio: {round(sum(is_within_stopping_criteria) / len(mean_obs_stats), 2)}')
        if all(is_within_stopping_criteria):
            return True, ste
        return False, ste

    @staticmethod
    def _no_func(val):
        return val

    @staticmethod
    def _round_to_int(val):
        return round(val)
