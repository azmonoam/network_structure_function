from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import random
from ergm_model.data_halper import get_list_of_functions_from_features_names
from ergm_model.ergm import Ergm
from ergm_model.ergm_utils import get_allowed_connections_based_on_ff_network_dims
from ergm_model.methods_random_dim import MethodsRandomDim
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from utils.main_utils import get_all_possible_dims
import copy


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

        self.allowed_connections = self.get_all_allowed_connections(
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
        )
        self.methods = MethodsRandomDim(
            num_neurons=n_nodes,
            output_size=output_size,
            possible_dims=list(self.allowed_connections.keys())
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
        if self.restrict_allowed_connections:
            for itter in range(seperation):
                for i, j in self.allowed_connections[dimensions]:
                    current_graph, current_stats, sum_delta_stats = self._update_ergm(
                        coefs=coefs,
                        current_graph=current_graph,
                        i=i,
                        j=j,
                        current_stats=current_stats,
                        sum_delta_stats=sum_delta_stats,
                        dimensions=dimensions,
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

    def get_all_allowed_connections(
            self,
            input_size: Optional[int] = None,
            output_size: Optional[int] = None,
            num_layers: Optional[int] = None,
    ) -> Dict[List[int], List[Tuple[int, int]]]:
        possible_dims = get_all_possible_dims(
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
            num_total_neurons=self.n_nodes,
        )
        return {
            tuple(dim): get_allowed_connections_based_on_ff_network_dims(
                dimensions=dim,
            )
            for dim in possible_dims
        }

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
            for g, dim in example_graphs:
                for i, j in self.allowed_connections[dim]:
                    train_sample, target_sample = self._get_state_sample(
                        graph=g,
                        i=i,
                        j=j,
                        dimensions=dim,
                    )
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
        dimensions = random.choice(list(self.allowed_connections.keys()))
        return current_graph, dimensions

    def _find_graphs_dim(
            self,
            graph: np.ndarray,
    ):
        rs, cs = np.nonzero(graph)
        s = set(
            (rs[i], cs[i])
            for i in range(rs.shape[0])
        )
        possible_dims = [
            possible_dim
            for possible_dim, possible_allowed_connections in self.allowed_connections_per_dim.items()
            if s.issubset(possible_allowed_connections)
        ]
        num_possible_dims = len(possible_dims)

        if num_possible_dims == 1:
            return possible_dims[0]
        if num_possible_dims == 0:
            return None
        return random.choice(possible_dims)