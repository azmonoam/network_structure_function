import subprocess
from copy import deepcopy
from typing import List, Tuple, Optional

import numpy as np
from joblib import Parallel, delayed, load
from sklearn.linear_model import LogisticRegression

from ergm_model.ergm_utils import switch_edge
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance


class Ergm:
    def __init__(
            self,
            features_names: List[str],
            find_feature_dist: Optional[FindFeaturesDistByPerformance] = None,
            target_mean_features: Optional[List[float]] = None,
            n_nodes: Optional[int] = None,
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
        self.n_nodes = n_nodes
        try:
            self.n_threads = int(subprocess.run('nproc', capture_output=True).stdout)
        except FileNotFoundError:
            self.n_threads = n_threads
        self.find_feature_dist = find_feature_dist

        self._init_target_features(
            target_mean_features=target_mean_features,
            num_isolated_neurones_feature=num_isolated_neurones_feature,
        )
        self.errors = errors
        self.seperation = seperation
        self.features_names = features_names
        if num_isolated_neurones_feature is not None:
            self.features_names.append('num_isolated_neurones')
            self.errors = np.append(self.errors, 0.0001)
        self.restrict_allowed_connections = restrict_allowed_connections

        self.all_mean_recorded_coeffs = []
        self.all_mean_obs_stats = []
        self.all_errors = []
        self.all_accepted_log_likelihoods = []

        self.num_graphs_to_sample_for_early_stopping = num_graphs_to_sample_for_early_stopping
        self.early_stopping_criteria, self.loose_early_stopping_criteria = self._get_stopping_params(
            early_stopping_type=early_stopping_type,
            distance_early_stopping_criteria_num_sigmas=distance_early_stopping_criteria_num_sigmas,
            mse_early_stopping_criteria_factor=mse_early_stopping_criteria_factor,
            num_graphs_to_sample_for_early_stopping=self.num_graphs_to_sample_for_early_stopping,
            num_stds=num_stds,
        )

    def _init_coeffs(
            self,
            initial_coeffs_values: Optional[List[float]] = None,
            example_graphs_path: Optional[str] = None,
            example_graphs: Optional[np.ndarray] = None,
    ):
        self.coefs = initial_coeffs_values
        if self.coefs is not None:
            return
        self.coefs, _ = self._get_initial_coeffs_values(
            example_graphs_path=example_graphs_path,
            example_graphs=example_graphs,
        )

    def _init_target_features(
            self,
            target_mean_features: Optional[List[float]],
            num_isolated_neurones_feature: Optional[bool],
    ):
        if target_mean_features:
            self.target_mean_features = target_mean_features
        elif self.find_feature_dist:
            self.calc_distance_from_target = self.find_feature_dist.calc_distance_from_mean
            self.target_mean_features = self.find_feature_dist.target_mean_features
        else:
            raise ValueError("Either 'find_feature_dist' or 'target_mean_features' need to be set")
        if num_isolated_neurones_feature is not None:
            self.target_mean_features.append(num_isolated_neurones_feature)

    def find_coefs(
            self,
            n_iterations: int = 10000,
            n_graph_samples: int = 75,
            start_alpha: float = 0.0005,
            end_alpha: float = 0.0001,
            burning: int = 1,
            stop_every: int = 15,
            test_for_loose: bool = True,
            num_tests_for_alphas: int = 100,
            last_changed_itter: int = 5000,
            use_cov_noise: bool = True,
            *args,
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
                        return recorded_coeffs, mean_recorded_coeff, graphs, graphs_stats
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
        return recorded_coeffs, mean_recorded_coeff, graphs, graphs_stats

    def sample_new_graph(
            self,
            coefs: np.ndarray,
            warmap: int = 0,
    ) -> Tuple[float, np.ndarray, List[float]]:
        pass

    def _update_ergm(
            self,
            coefs: np.ndarray,
            current_graph: np.ndarray,
            i: int,
            j: int,
            current_stats: List[float],
            sum_delta_stats: float,
            **kwargs,
    ):
        new_graph = switch_edge(
            graph=current_graph,
            i=i,
            j=j,
        )
        new_stats = self.calc_stats(
            graph=new_graph,
            **kwargs,
        )
        delta_stats = np.subtract(new_stats, current_stats)
        prob = np.exp(np.dot(coefs, delta_stats))
        if np.random.rand() < min(1, prob):
            return new_graph, new_stats, np.add(sum_delta_stats, delta_stats)
        return current_graph, current_stats, sum_delta_stats

    def compute_log_likelihood(
            self,
            sum_delta_stats,
            last_accepted_coefs,
    ) -> float:
        ll = np.dot(self.target_mean_features, self.coefs) - np.dot(self.target_mean_features, last_accepted_coefs)
        ll += np.dot(sum_delta_stats, last_accepted_coefs) - np.dot(sum_delta_stats, self.coefs)
        # ll += stats.norm.logpdf(self.coefs, loc=0, scale=1)[0] - stats.norm.logpdf(last_accepted_coefs, loc=0, scale=1)[0]
        return ll

    def sample_multiple(
            self,
            coefs: np.ndarray,
            num_graphs_to_sample: int = 200,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        res = self.sample_multiple_parallel_wrapper(
            coefs=coefs,
            num_graphs_to_sample=num_graphs_to_sample,
        )
        g_stats = []
        graphs = []
        for _, current_graph, current_stats in res:
            g_stats.append(current_stats)
            graphs.append(current_graph)
        return graphs, g_stats

    def sample_multiple_parallel_wrapper(
            self,
            coefs: np.ndarray,
            num_graphs_to_sample: int = 200,
    ):
        return Parallel(
            n_jobs=self.n_threads,
            timeout=99999,
        )(
            delayed
            (self.sample_new_graph)(
                seperation=5,
                coefs=coefs,
            )
            for _ in range(num_graphs_to_sample)
        )

    def calc_stats(
            self,
            graph,
            **kwargs,
    ) -> List[float]:
        return [
            f(graph, f_name, **kwargs)
            for f, f_name in zip(self.features, self.features_names)
        ]

    def calc_mse(
            self,
            mean_obs_stats,
            early_stopping_criteria,
    ) -> Tuple[bool, List[float]]:
        mses = [
            (mean_obs_stat - target) ** 2 / target
            for mean_obs_stat, target in zip(mean_obs_stats, self.target_mean_features)
        ]
        is_within_stopping_criteria = [
            mse < limit
            for mse, limit in zip(mses, early_stopping_criteria)
        ]
        print(f'mses: {mses}')
        print(f'is good enough ratio: {round(sum(is_within_stopping_criteria) / len(mean_obs_stats), 2)}')
        if all(is_within_stopping_criteria):
            return True, mses
        return False, mses

    def calc_me(
            self,
            mean_obs_stats,
            early_stopping_criteria,
    ) -> Tuple[bool, List[float]]:
        mses = [
            ((mean_obs_stat - target) ** 2) ** 0.5
            for mean_obs_stat, target in zip(mean_obs_stats, self.target_mean_features)
        ]
        is_within_stopping_criteria = [
            mse < limit
            for mse, limit in zip(mses, early_stopping_criteria)
        ]
        print(f'error: {mses}')
        print(f'sum error: {sum(mses)}')
        print(f'is good enough ratio: {round(sum(is_within_stopping_criteria) / len(mean_obs_stats), 2)}')
        if all(is_within_stopping_criteria):
            return True, mses
        return False, mses

    def calc_error_percentage_of_std(
            self,
            mean_obs_stats,
            early_stopping_criteria,
    ):
        ste = [
            abs(mean_obs_stat - target) / std
            for mean_obs_stat, target, std
            in zip(mean_obs_stats, self.target_mean_features, self.errors)
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

    def _find_error_bases_early_stopping_criteria(
            self,
            num_graphs_to_sample_for_early_stopping: int,
    ) -> np.ndarray:
        if self.errors is not None:
            return self.errors
        if self.find_feature_dist is None:
            raise ValueError("'find_feature_dist' must be set to use 'distance' method")
        means = np.zeros((num_graphs_to_sample_for_early_stopping, len(self.features_names)))
        for i in range(num_graphs_to_sample_for_early_stopping):
            means[i] = np.mean(
                self.find_feature_dist.samples_to_model.sample(num_graphs_to_sample_for_early_stopping, axis=0),
                axis=0)
        return np.std(means, axis=0)

    def calc_distance_from_mean(
            self,
            mean_obs_stats,
            early_stopping_criteria,
    ) -> Tuple[bool, float]:
        if self.find_feature_dist is None:
            raise ValueError("'find_feature_dist' must be set to use 'distance' method")
        distance = self.find_feature_dist.calc_distance_from_mean(
            sample=np.array(mean_obs_stats)
        )
        print(f'mean_obs_stats: {mean_obs_stats}')
        print(f'distance: {distance}')
        if distance < early_stopping_criteria:
            return True, distance
        return False, distance

    def no_stopping(
            self,
            mean_obs_stats,
            early_stopping_criteria,
    ) -> Tuple[bool, float]:
        pass

    def _remove_modularity(self):
        if 'modularity' not in self.features_names:
            return
        modularity_index = self.features_names.index('modularity')
        self.features_names.pop(modularity_index)
        self.target_mean_features.pop(modularity_index)
        self.features.pop(modularity_index)
        if self.find_feature_dist is not None:
            self.find_feature_dist.means.pop(modularity_index)
            self.find_feature_dist.covs.drop(labels=modularity_index, axis=0, inplace=True)
            self.find_feature_dist.covs.drop(labels=modularity_index, axis=1, inplace=True)
            self.find_feature_dist.inv_covs = np.linalg.inv(self.find_feature_dist.covs.to_numpy())

    def _get_stopping_params(
            self,
            early_stopping_type: str,
            mse_early_stopping_criteria_factor: float,
            distance_early_stopping_criteria_num_sigmas: float,
            num_graphs_to_sample_for_early_stopping: int,
            num_stds: int,
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if early_stopping_type == 'mse':
            early_stopping_criteria = [
                mse_early_stopping_criteria_factor
                for _ in self.target_mean_features
            ]
            loose_early_stopping_criteria = [e * 2 for e in early_stopping_criteria]
            self.stopping_function = self.calc_mse
        elif early_stopping_type == 'distance':
            early_stopping_criteria = distance_early_stopping_criteria_num_sigmas
            loose_early_stopping_criteria = distance_early_stopping_criteria_num_sigmas * 2
            self.stopping_function = self.calc_distance_from_mean
        elif early_stopping_type == 'error':
            early_stopping_criteria = self._find_error_bases_early_stopping_criteria(
                num_graphs_to_sample_for_early_stopping=num_graphs_to_sample_for_early_stopping,
            )
            loose_early_stopping_criteria = [e * 2 for e in early_stopping_criteria]
            self.stopping_function = self.calc_me
        elif early_stopping_type == 'std':
            early_stopping_criteria = [
                1 * num_stds
                for _ in self.target_mean_features
            ]
            loose_early_stopping_criteria = [e * 2 for e in early_stopping_criteria]
            self.stopping_function = self.calc_error_percentage_of_std
        else:
            self.stopping_function = self.no_stopping
            early_stopping_criteria = None
            loose_early_stopping_criteria = None
        print(f"early_stopping_criteria: {[round(e, 8) for e in early_stopping_criteria]}")
        return early_stopping_criteria, loose_early_stopping_criteria

    def _get_initial_coeffs_values(
            self,
            example_graphs_path: Optional[str],
            example_graphs: Optional[np.ndarray],
    ) -> Tuple[List[float], Optional[np.ndarray]]:
        if example_graphs is None:
            if example_graphs_path is None:
                return [0] * len(self.features), None
            with open(example_graphs_path, 'rb') as fp:
                example_graphs = load(fp)
        example_graphs_stats = [
            self.calc_stats(
                graph=g,
            )
            for g in example_graphs
        ]
        print(f'mean example graphs stats: {np.mean(example_graphs_stats, axis=0)}')
        train, target = self._get_train_and_target_from_example_graphs(
            example_graphs=example_graphs,
        )
        logistic_regression_coeffs = self._get_logistic_regression_coeffs(
            train=train,
            target=target,
        )
        return logistic_regression_coeffs, example_graphs

    def _get_train_and_target_from_example_graphs(
            self,
            example_graphs: Optional[np.ndarray],
    ):
        train, target = [], []
        if self.restrict_allowed_connections:
            for g in example_graphs:
                for i, j in self.allowed_connections:
                    train_sample, target_sample = self._get_state_sample(
                        graph=g,
                        i=i,
                        j=j,
                    )
                    train.append(train_sample)
                    target.append(target_sample)
        else:
            for g in example_graphs:
                for i in range(1, self.n_nodes):
                    for j in range(i):
                        train_sample, target_sample = self._get_state_sample(
                            graph=g,
                            i=i,
                            j=j,
                        )
                        train.append(train_sample)
                        target.append(target_sample)
        return train, target

    @staticmethod
    def _get_logistic_regression_coeffs(
            train,
            target,
    ):
        clf = LogisticRegression(C=0.1, fit_intercept=False, random_state=42)
        clf.fit(train, target)
        return clf.coef_[0]

    def _get_state_sample(
            self,
            graph: np.ndarray,
            i: int,
            j: int,
            **kwargs,
    ) -> Tuple[List[float], int]:
        target_sample = graph[i, j]
        x_0, x_1 = deepcopy(graph), deepcopy(graph)
        x_0[i, j] = 0
        x_1[i, j] = 1
        graph_stats_0 = self.calc_stats(
            graph=x_0,
            **kwargs,
        )
        graph_stats_1 = self.calc_stats(
            graph=x_1,
            **kwargs,
        )
        train_sample = np.subtract(graph_stats_1, graph_stats_0)
        return train_sample, target_sample

    @staticmethod
    def _calc_mean_recorded_coeff(
            recorded_coeffs: List[List[float]],
            burning: int,
    ) -> np.ndarray:
        mean_recorded_coeff = np.mean(recorded_coeffs[burning:], axis=0)
        print(f'mean_recorded_coeff: {mean_recorded_coeff}')
        return mean_recorded_coeff

    def _calc_early_stopping(
            self,
            mean_recorded_coeff: np.ndarray,
            num_graphs_to_sample_for_early_stopping: int,
    ) -> Tuple[List[np.ndarray], List[List[float]], np.ndarray, bool, List[float]]:
        graphs, graphs_stats = self.sample_multiple(
            coefs=mean_recorded_coeff,
            num_graphs_to_sample=num_graphs_to_sample_for_early_stopping,
        )
        mean_obs_stats = np.mean(graphs_stats, axis=0)
        print(f'mean obs stats: {mean_obs_stats}, std obs stats: {np.std(graphs_stats, axis=0)}')
        is_within_stopping_caretria, stopping_value = self.stopping_function(
            mean_obs_stats,
            self.early_stopping_criteria,
        )
        return graphs, graphs_stats, mean_obs_stats, is_within_stopping_caretria, stopping_value

    def _test_early_stopping(
            self,
            mean_recorded_coeff: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[List[float]], np.ndarray, bool, List[float]]:
        graphs, graphs_stats, mean_obs_stats, is_within_stopping_caretria, stopping_value = self._calc_early_stopping(
            mean_recorded_coeff=mean_recorded_coeff,
            num_graphs_to_sample_for_early_stopping=self.num_graphs_to_sample_for_early_stopping,
        )
        if is_within_stopping_caretria:
            increased_num_graphs_to_sample = self.num_graphs_to_sample_for_early_stopping * 5
            print(f'sampling {self.num_graphs_to_sample_for_early_stopping} graphs converged to required values! '
                  f'sampling {increased_num_graphs_to_sample} graphs...')
            graphs2, graphs_stats2, mean_obs_stats, is_within_stopping_caretria, stopping_value = self._calc_early_stopping(
                mean_recorded_coeff=mean_recorded_coeff,
                num_graphs_to_sample_for_early_stopping=increased_num_graphs_to_sample,
            )
            graphs += graphs2
            graphs_stats += graphs_stats2
            if is_within_stopping_caretria:
                mean_obs_stats = np.mean(graphs_stats, axis=0)
                print(f'combined samples - mean obs stats: {mean_obs_stats}, std obs stats: {np.std(graphs_stats, axis=0)}')
                is_within_stopping_caretria, stopping_value = self.stopping_function(
                    mean_obs_stats,
                    self.early_stopping_criteria,
                )
        return graphs, graphs_stats, mean_obs_stats, is_within_stopping_caretria, stopping_value

    def _is_within_loose_early_stopping_criteria(
            self,
            stopping_value: List[float],
    ):
        loose_is_within_stopping_criteria = [
            mse < limit
            for mse, limit in zip(stopping_value, self.loose_early_stopping_criteria)
        ]
        return round(sum(loose_is_within_stopping_criteria) / len(self.loose_early_stopping_criteria), 2) >= 0.8

    @staticmethod
    def _update_alphas_and_stopping(
            alpha: np.ndarray,
            stop_every: int,
            itter: int
    ):
        print(f'cuurent stopping interval: {stop_every}, next alphas: {alpha[:, itter + 1]}')
        stop_every = max(stop_every // 1.2, 2)
        alpha = alpha / 1.2 * np.ones_like(alpha)
        print(f'updated stopping interval: {stop_every}, next alphas: {alpha[:, itter + 1]}')
        return stop_every, alpha

    def _test_and_update_alphas(
            self,
            itter: int,
            alpha: np.array,
            last_changed_itter: int,
            num_tests: int,
            start_alpha: float,
            diff_prec: float = 1.1,
    ):
        if len(self.all_errors) < 2 * num_tests:
            return alpha, last_changed_itter
        sums = np.sum(self.all_errors, axis=1)
        current_mean_error = np.mean(sums[-num_tests:])
        prev_mean_error = np.mean(sums[-2 * num_tests:-num_tests])
        if current_mean_error * diff_prec > prev_mean_error:
            cuurent_a = alpha[:, itter]
            next_alpha = alpha * 2 * np.ones_like(alpha)
            if next_alpha[:, itter][0] < start_alpha * 3:
                print(
                    f"sum errors didn't improve by {diff_prec} in the last {num_tests} calculations, updated next alphas: "
                    f"{cuurent_a} to {next_alpha[:, itter]}")
                return next_alpha, itter
        return alpha, last_changed_itter
