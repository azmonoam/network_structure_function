from typing import List, Tuple, Optional
from typing import Union

import numpy as np
from joblib import Parallel, delayed

from ergm_model.ergm_from_reference_graph import ErgmReferenceGraph
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance


class ErgmReferenceGraphMultiObs(ErgmReferenceGraph):
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
            seperation: int = 3,
    ):
        super().__init__(features_names, find_feature_dist, target_mean_features, dimensions, n_nodes,
                         example_graphs_path, example_graphs, initial_coeffs_values, restrict_allowed_connections,
                         errors, n_threads, num_isolated_neurones_feature, distance_early_stopping_criteria_num_sigmas,
                         num_graphs_to_sample_for_early_stopping, mse_early_stopping_criteria_factor,
                         early_stopping_type, num_stds, num_example_graphs, seperation)

    def find_coefs(
            self,
            early_stopping_type: str = 'distance',
            n_iterations: int = 10000,
            n_graph_samples: int = 15,
            start_alpha: float = 0.0005,
            end_alpha: float = 0.0001,
            burning: int = 1,
            stop_every: int = 15,
            test_for_loose: bool = True,
            num_tests_for_alphas: int = 100,
            last_changed_itter: int = 5000,
            use_cov_noise: bool = True,
    ):
        last_accepted_coefs = self.coefs
        last_accepted_log_likelihood = 0
        recorded_coeffs = []
        I = np.eye(len(self.coefs))
        alpha = np.array(
            [
                np.linspace(start_alpha, end_alpha, n_iterations)
                for _ in self.coefs
            ]
        )
        for itter in range(n_iterations):
            for _ in range(n_graph_samples):
                if use_cov_noise and len(self.coefs) > 1 and itter > 100:
                    try:
                        noise = alpha[:, itter] * np.cov(recorded_coeffs, rowvar=False)
                    except:
                        noise = I * alpha[:, itter]
                else:
                    noise = I * alpha[:, itter]
                self.coefs = np.random.multivariate_normal(last_accepted_coefs, noise)
                sum_delta_stats, _, _ = self.sample_new_graph(
                    coefs=self.coefs,
                )
                log_likelihood = self.compute_log_likelihood(
                    sum_delta_stats,
                    last_accepted_coefs,
                )
                acceptence_rate = np.log(np.random.rand())
                if acceptence_rate < log_likelihood:
                    last_accepted_coefs = self.coefs
                    last_accepted_log_likelihood = log_likelihood
            self.all_accepted_log_likelihoods.append(last_accepted_log_likelihood)
            recorded_coeffs.append(last_accepted_coefs)
            if itter % stop_every == 0 and itter >= burning:
                print(f'\nitter: {itter}/{n_iterations}')
                mean_recorded_coeff = self._calc_mean_recorded_coeff(
                    recorded_coeffs=recorded_coeffs,
                    burning=burning,
                )
                if self.early_stopping_criteria is not None:
                    graphs, graphs_stats, mean_obs_stats, is_within_stopping_caretria, stopping_value = self._test_early_stopping(
                        mean_recorded_coeff=mean_recorded_coeff,
                    )
                    self.all_mean_recorded_coeffs.append(mean_recorded_coeff)
                    self.all_mean_obs_stats.append(mean_obs_stats)
                    self.all_errors.append(stopping_value)
                    if is_within_stopping_caretria:
                        return recorded_coeffs, mean_recorded_coeff, graphs, graphs_stats, True
                    if test_for_loose:
                        if self._is_within_loose_early_stopping_criteria(
                                stopping_value=stopping_value,
                        ):
                            stop_every, alpha = self._update_alphas_and_stopping(
                                stop_every=stop_every,
                                alpha=alpha,
                                itter=itter,
                            )
                        elif itter > last_changed_itter + (2 * num_tests_for_alphas):
                            alpha, last_changed_itter = self._test_and_update_alphas(
                                alpha=alpha,
                                itter=itter,
                                last_changed_itter=last_changed_itter,
                                num_tests=num_tests_for_alphas,
                                start_alpha=start_alpha,
                            )
                    elif itter > last_changed_itter + (2 * num_tests_for_alphas):
                        alpha, last_changed_itter = self._test_and_update_alphas(
                            alpha=alpha,
                            itter=itter,
                            last_changed_itter=last_changed_itter,
                            num_tests=num_tests_for_alphas,
                            start_alpha=start_alpha,
                        )
        return recorded_coeffs, mean_recorded_coeff, graphs, graphs_stats, False

    def sample_new_graph(
            self,
            coefs: np.ndarray,
            warmap: int = 0,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        all_graphs = []
        all_stats = []
        sum_delta_stats = 0
        all_sampled_graphs = self.sample_example_graphs_wrapper(
            coefs=coefs,
            sum_delta_stats=sum_delta_stats
        )
        for current_graph, current_stats in all_sampled_graphs:
            all_graphs.append(current_graph)
            all_stats.append(current_stats)
        sum_all_delta_stats = 1 / len(self.example_graphs) * np.sum(all_stats, axis=0)
        return sum_all_delta_stats, all_graphs, all_stats

    def sample_single_graph(
            self,
            coefs: np.ndarray,
            current_graph: np.ndarray,
            sum_delta_stats: int = 0,
    ):
        current_stats = self.calc_stats(
            graph=current_graph,
        )
        for itter in range(self.seperation):
            for i, j in self.allowed_connections:
                current_graph, current_stats, _ = self._update_ergm(
                    coefs=coefs,
                    current_graph=current_graph,
                    i=i,
                    j=j,
                    current_stats=current_stats,
                    sum_delta_stats=sum_delta_stats,
                )
        return current_graph, current_stats

    def sample_example_graphs_wrapper(
            self,
            coefs: np.ndarray,
            sum_delta_stats: int = 0,
    ):
        return Parallel(
            n_jobs=self.n_threads,
            timeout=99999,
        )(
            delayed
            (self.sample_single_graph)(
                coefs=coefs,
                current_graph=current_graph,
                sum_delta_stats=sum_delta_stats,
            )
            for current_graph in self.example_graphs
        )

    def sample_multiple(
            self,
            coefs: np.ndarray,
            num_graphs_to_sample: int = 200,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:

        g_stats = []
        graphs = []
        for _ in range(num_graphs_to_sample):
            _, current_graph, current_stats = self.sample_new_graph(
                coefs=coefs,
            )
            g_stats += current_stats
            graphs += current_graph
        return graphs, g_stats

    @staticmethod
    def _update_alphas_and_stopping(
            alpha: np.ndarray,
            stop_every: int,
            itter: int,
    ):
        print(f'cuurent stopping interval: {stop_every}, next alphas: {alpha[:, itter + 1]}')
        stop_every = max(stop_every // 1.2, 2)
        end_alpha = alpha[0, -1]
        start_alpha = max(alpha[0, 0] / 1.2, end_alpha)
        alpha = np.array(
            [
                np.linspace(start_alpha, end_alpha, alpha.shape[1])
                for _ in range(alpha.shape[0])
            ]
        )
        print(f'updated stopping interval: {stop_every}, next alphas: {alpha[:, itter + 1]}')
        return stop_every, alpha
