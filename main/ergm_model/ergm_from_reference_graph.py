import copy
import random
from typing import List, Tuple, Optional, Union

import numpy as np
from joblib import load

from ergm_model.ergm_classic import ErgmClassic
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from parameters.general_paramters import RANDOM_SEED

class ErgmReferenceGraph(ErgmClassic):
    def __init__(
            self,
            features_names: List[str],
            find_feature_dist: Optional[FindFeaturesDistByPerformance] = None,
            target_mean_features: Optional[List[float]] = None,
            dimensions: Optional[Union[Tuple[int, ...], List[int]]] = None,
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
            num_example_graphs: int = 100,
            seperation: int = 5
    ):
        super().__init__(features_names, find_feature_dist, target_mean_features, dimensions, n_nodes,
                         example_graphs_path, example_graphs, initial_coeffs_values, restrict_allowed_connections,
                         errors, n_threads, num_isolated_neurones_feature, distance_early_stopping_criteria_num_sigmas,
                         num_graphs_to_sample_for_early_stopping, mse_early_stopping_criteria_factor,
                         early_stopping_type, num_stds, seperation
                         )
        random.seed = RANDOM_SEED
        self._choose_initiation_graphs(
            num_example_graphs,
        )

    def _init_coeffs(
            self,
            initial_coeffs_values: Optional[List[float]] = None,
            example_graphs_path: Optional[str] = None,
            example_graphs: Optional[np.ndarray] = None,
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

    def sample_new_graph(
            self,
            coefs: np.ndarray,
            warmap: int = 0,
    ) -> Tuple[float, np.ndarray, List[float]]:
        current_graph = random.choice(self.example_graphs)
        current_stats = self.calc_stats(
            graph=current_graph,
        )
        sum_delta_stats = copy.deepcopy(current_stats)
        if self.restrict_allowed_connections:
            for itter in range(self.seperation):
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

    def _get_example_graphs(
            self,
            example_graphs_path: Optional[str],
    ) -> Optional[np.ndarray]:
        if example_graphs_path is None:
            return None
        with open(example_graphs_path, 'rb') as fp:
            example_graphs = load(fp)
        return example_graphs

    def _choose_initiation_graphs(
            self,
            num_graphs: int,
    ):
        if len(self.example_graphs) < num_graphs:
            return
        i = 0
        print(f"Choosing {num_graphs} example graphs...")
        while i < len(self.example_graphs):
            chosen = random.sample(self.example_graphs, num_graphs)
            chosen_stats = [
                self.calc_stats(
                    graph=g,
                )
                for g in chosen
            ]
            is_within_stopping_caretria, chosen_g_errors = self.stopping_function(
                np.mean(chosen_stats, axis=0),
                self.early_stopping_criteria,
            )
            if sum(1 for error in chosen_g_errors if error < 1) == len(self.features):
                self.example_graphs = chosen
                return
            i += 1
        raise ValueError(f"could not find {num_graphs} with the correct mean")
