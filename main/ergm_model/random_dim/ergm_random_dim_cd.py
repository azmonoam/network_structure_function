import random
from typing import List, Tuple, Optional, Union

import numpy as np

from ergm_model.random_dim.ergm_from_reference_graph_random_dim import ErgmReferenceGraphRandomDim
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance


class ErgmRandomdimCD(ErgmReferenceGraphRandomDim):
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
                         num_layers, example_graphs_path, example_graphs,
                         initial_coeffs_values, restrict_allowed_connections, errors, n_threads,
                         num_isolated_neurones_feature)

    def re_init_coeff_values(
            self,
            n_iterations: int = 60,
            n_graph_samples: int = 75,
            start_alpha: float = 0.0005,
            end_alpha: float = 0.0001,
            k: int = 16,
            s: int = 500,
    ):
        last_accepted_coefs = self.coefs
        I = np.eye(len(self.coefs))
        alpha = np.array(
            [
                np.linspace(start_alpha, end_alpha, n_iterations)
                for _ in self.coefs
            ]
        )
        noise = I * alpha[:, 0]
        for itter in range(n_iterations):
            sum_delta_stats, all_stats = self.sample_new_graph_cd(
                k=k,
                coefs=self.coefs,
                s=s,
            )
            log_likelihood = self.compute_log_likelihood(
                sum_delta_stats,
                last_accepted_coefs,
            )
            acceptence_rate = np.log(np.random.rand())
            if acceptence_rate < log_likelihood:
                last_accepted_coefs = self.coefs
            noise = np.mean(all_stats, axis=0) / ((1 / len(all_stats)) * np.sum(np.var(all_stats, axis=1)))
            self.coefs = last_accepted_coefs - noise
        self.coefs = last_accepted_coefs
        print(f"CD coeffs: {self.coefs}")

    def sample_new_graph_cd(
            self,
            coefs: np.ndarray,
            k: int,
            s: int,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        graph_samples = random.sample(self.example_graphs, s)
        all_stats = []
        for current_graph, dimensions in graph_samples:
            current_stats = self.calc_stats(
                graph=current_graph,
                dimensions=dimensions
            )
            connections_to_itter_over = self.allowed_connections.copy()
            random.shuffle(connections_to_itter_over)
            sum_delta_stats = np.zeros_like(current_stats)
            for i, j in connections_to_itter_over[:k]:
                current_graph, current_stats, sum_delta_stats = self._update_ergm(
                    coefs=coefs,
                    current_graph=current_graph,
                    i=i,
                    j=j,
                    current_stats=current_stats,
                    sum_delta_stats=sum_delta_stats,
                )
            all_stats.append(sum_delta_stats)
        sum_all_delta_stats = 1 / s * np.sum(all_stats, axis=0)
        return sum_all_delta_stats, all_stats

    def compute_log_likelihood(
            self,
            sum_delta_stats,
            last_accepted_coefs,
    ) -> float:
        ll = np.dot(sum_delta_stats, self.coefs) - np.dot(sum_delta_stats, last_accepted_coefs)
        return ll
