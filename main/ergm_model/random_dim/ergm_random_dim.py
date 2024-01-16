import copy
import itertools
import random
from typing import List, Tuple, Optional, Union

import numpy as np

from ergm_model.data_halper import get_list_of_functions_from_features_names
from ergm_model.ergm import Ergm
from ergm_model.ergm_utils import get_allowed_connections_based_on_ff_network_dims, creat_ff_fully_connected_mask
from ergm_model.ergm_utils import switch_edge
from ergm_model.methods_random_dim import MethodsRandomDim
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from utils.main_utils import get_all_possible_dims

random.seed = 94

class ErgmRandomdim(Ergm):
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
        super().__init__(features_names, find_feature_dist, target_mean_features, n_nodes,
                         restrict_allowed_connections, errors, n_threads, num_isolated_neurones_feature)

        self.n_nodes = n_nodes

        self._init_all_allowed_connections(
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
        )
        self.methods = MethodsRandomDim(
            num_neurons=n_nodes,
            output_size=output_size,
            possible_dims=self.allowed_dims,
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
            seperation: int,
            coefs: np.ndarray,
            warmap: int = 0,
    ) -> Tuple[float, np.ndarray, List[float]]:
        current_graph, dimensions = self._get_initial_graph_and_dim()
        current_stats = self.calc_stats(
            graph=current_graph,
            dimensions=dimensions
        )
        sum_delta_stats = copy.deepcopy(current_stats)
        connections_to_itter_over = self.allowed_connections.copy()
        if self.restrict_allowed_connections:
            for itter in range(seperation):
                random.shuffle(connections_to_itter_over)
                for i, j in connections_to_itter_over:
                    current_graph, current_stats, sum_delta_stats = self._update_ergm(
                        coefs=coefs,
                        current_graph=current_graph,
                        i=i,
                        j=j,
                        current_stats=current_stats,
                        sum_delta_stats=sum_delta_stats,
                    )
        else:
            for itter in range(seperation):
                for i in range(self.n_nodes):
                    for j in range(i + 1, self.n_nodes):
                        current_graph, current_stats, sum_delta_stats = self._update_ergm(
                            coefs=coefs,
                            current_graph=current_graph,
                            i=i,
                            j=j,
                            current_stats=current_stats,
                            sum_delta_stats=sum_delta_stats,
                            dimensions=dimensions,
                        )
        return sum_delta_stats, current_graph, current_stats

    def _init_all_allowed_connections(
            self,
            input_size: Optional[int] = None,
            output_size: Optional[int] = None,
            num_layers: Optional[int] = None,
    ):
        self.allowed_dims = get_all_possible_dims(
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
            num_total_neurons=self.n_nodes,
        )
        allowed_connections_per_dim = [
            set(
                get_allowed_connections_based_on_ff_network_dims(
                    dimensions=dim,
                )
            )
            for dim in self.allowed_dims
        ]
        self.allowed_connections = list(set(itertools.chain(*allowed_connections_per_dim)))
        self.all_ff_graphs = np.zeros((len(self.allowed_dims), self.n_nodes, self.n_nodes))
        for i, dim in enumerate(self.allowed_dims):
            self.all_ff_graphs[i] = creat_ff_fully_connected_mask(
                num_neurons=self.n_nodes,
                dimensions=dim,
            ) * -1

    def calc_stats(
            self,
            graph,
            dimensions: Optional[List[int]] = None
    ) -> List[float]:
        return [
            f(graph, f_name, dimensions)
            for f, f_name in zip(self.features, self.features_names)
        ]

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
        if self.restrict_allowed_connections:
            for g, _ in example_graphs:
                connections_to_itter_over = self.allowed_connections.copy()
                random.shuffle(connections_to_itter_over)
                for i, j in connections_to_itter_over:
                    state_sample = self._get_state_sample(
                        graph=g,
                        i=i,
                        j=j,
                    )
                    if state_sample is not None:
                        train_sample, target_sample = state_sample
                        train.append(train_sample)
                        target.append(target_sample)
        else:
            for g, dim in example_graphs:
                for i in range(1, self.n_nodes):
                    for j in range(i):
                        train_sample, target_sample = self._get_state_sample(
                            graph=g,
                            i=i,
                            j=j,
                            dimensions=dim,
                        )
                        train.append(train_sample)
                        target.append(target_sample)
        return train, target

    def _get_initial_graph_and_dim(self):
        current_graph = np.zeros((self.n_nodes, self.n_nodes))
        dimensions = random.choice(self.allowed_connections)
        return current_graph, dimensions

    def _get_state_sample(
            self,
            graph: np.ndarray,
            i: int,
            j: int,
            **kwargs,
    ) -> Optional[Tuple[List[float], int]]:
        target_sample = graph[i, j]
        x_0, x_1 = copy.deepcopy(graph), copy.deepcopy(graph)
        x_0[i, j] = 0
        x_1[i, j] = 1
        x_0_dim = self._find_graphs_dim(
            graph=x_0,
        )
        if x_0_dim is None:
            return
        x_1_dim = self._find_graphs_dim(
            graph=x_1,
        )
        if x_1_dim is None:
            return
        graph_stats_0 = self.calc_stats(
            graph=x_0,
            dimensions=x_0_dim
        )
        graph_stats_1 = self.calc_stats(
            graph=x_1,
            dimensions=x_1_dim
        )
        train_sample = np.subtract(graph_stats_1, graph_stats_0)
        return train_sample, target_sample

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
        dim = self._find_graphs_dim(
            graph=new_graph,
        )
        if dim is None:
            return current_graph, current_stats, sum_delta_stats
        new_stats = self.calc_stats(
            graph=new_graph,
            dimensions=dim,
        )
        delta_stats = np.subtract(new_stats, current_stats)
        prob = np.exp(np.dot(coefs, delta_stats))
        if np.random.rand() < min(1, prob):
            return new_graph, new_stats, np.add(sum_delta_stats, delta_stats)
        return current_graph, current_stats, sum_delta_stats

    def _find_graphs_dim(
            self,
            graph: np.ndarray,
    ):
        test_array = self.all_ff_graphs + graph
        possible_dims_inds = np.where(np.sum(np.sum(np.isin(test_array, [1]), axis=1), axis=1) == 0)[0]
        num_possible_dims = possible_dims_inds.shape[0]

        if num_possible_dims == 1:
            return self.allowed_dims[possible_dims_inds.item()]
        if num_possible_dims == 0:
            return None
        return self.allowed_dims[np.random.choice(possible_dims_inds).item()]
