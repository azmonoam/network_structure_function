import copy
import itertools
from typing import List, Tuple, Optional, Union

import numpy as np

from ergm_model.random_dim.ergm_from_reference_graph_random_dim import ErgmReferenceGraphRandomDim
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance


class ErgmRandomdimMCLE(ErgmReferenceGraphRandomDim):
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

    def _get_train_and_target_from_example_graphs(
            self,
            example_graphs: Optional[List[List[Union[np.ndarray, Tuple[int, ...]]]]],
    ):
        example_graphs_stats = [
            self.calc_stats(
                graph=g,
                dimensions=dim,
            )
            for g, dim in example_graphs
        ]
        print(f'mean example graphs stats: {np.mean(example_graphs_stats, axis=0)}')
        train, target = [], []
        connections_to_itter_over = itertools.combinations(self.allowed_connections, 2)
        for g, dim in example_graphs:
            for (i, j), (k, l) in connections_to_itter_over:
                state_sample = self._get_state_sample(
                    graph=g,
                    i=i,
                    j=j,
                    k=k,
                    l=l,
                    orignal_graph_dim=dim,
                )
                if state_sample is not None:
                    train_sample, target_sample = state_sample
                    train.append(train_sample)
                    target.append(target_sample)
        return train, target

    def _get_state_sample(
            self,
            graph: np.ndarray,
            i: int,
            j: int,
            k: int = None,
            l: int = None,
            orignal_graph_dim=None,
            **kwargs,
    ) -> Optional[Tuple[List[float], int]]:
        sum_graph_stats = np.zeros(len(self.features))
        possible_comb = [[0, 1], [1, 0], [1, 1], [0, 0]]
        classes = [0, 1, 1, 2]
        target_sample = (graph[i, j], graph[k, l])
        xs = [copy.deepcopy(graph), copy.deepcopy(graph), copy.deepcopy(graph), ]
        m = 0
        for ind, (v1, v2) in enumerate(possible_comb):
            if (v1, v2) == target_sample:
                c = classes[ind]
            else:
                x = xs[m]
                x[i, j] = v1
                x[k, l] = v2
                dim = self._find_graphs_dim(
                    graph=x,
                )
                if dim is not None:
                    graph_stats = self.calc_stats(
                        graph=x,
                        dimensions=dim
                    )
                    sum_graph_stats = sum_graph_stats + graph_stats
                m += 1
        if sum(sum_graph_stats) == 0:
            return None
        org_graph_stats = self.calc_stats(
            graph=graph,
            dimensions=orignal_graph_dim
        )
        train_sample = np.subtract(org_graph_stats, sum_graph_stats)
        return train_sample, c
