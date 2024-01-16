from typing import List, Optional, Callable, Dict, Set
import torch
import numpy as np

from scipy.special import comb

from nn_models.nn_model import NeuralNetModel
from nn_models.nn_model_five_layers import FeedforwardNeuralNetModelFiveLayers
from nn_models.nn_model_four_layers import FeedforwardNeuralNetModelFourLayers
from nn_models.nn_model_one_layer import FeedforwardNeuralNetModelOneLayer
from nn_models.nn_model_six_layers import FeedforwardNeuralNetModelSixLayers
from nn_models.nn_model_three_layers import FeedforwardNeuralNetModelThreeLayers
from nn_models.nn_model_two_layers import FeedforwardNeuralNetModelTwoLayers
from parameters.general_paramters import (
    general_allowed_bias_values,
    general_allowed_weights_values,
)
from parameters.general_paramters import motifs_full_name
from tasks import Task
from itertools import permutations


class BaseParams:
    start_dimensions: List[int]
    num_layers: int
    task_version_name: Optional[str]
    task_base_folder_name: Optional[str]
    task_global_name: str

    rule: Callable
    communities_input_symmetry: int
    input_dim: int
    task: Task
    mutation_probabilities: Dict[str, float]

    learning_rate: float
    num_epochs: int
    activate: str

    structural_features_vec_length: Optional[int] = None
    EDGE_MAPPING: Optional[List[str]] = None
    structural_features_full_name_vec: Optional[List[str]] = None
    structural_features_name_vec: Optional[List[str]] = None
    target_label_ranges: Optional[List[str]] = None

    allowed_weights_values: List[int] = general_allowed_weights_values
    allowed_bias_values: List[int] = general_allowed_bias_values

    base_path: str = '/home/labs/schneidmann/noamaz/modularity'
    flatten: bool
    use_bias: bool = True
    possible_middle_dims_options: Optional[List[List[int]]] = None

    def __init__(
            self,
            start_dimensions: Optional[List[int]] = None
    ):
        self.start_dimensions = start_dimensions
        if self.start_dimensions is None:
            self.start_dimensions = self._get_start_dimensions()
        self.max_possible_connections = self._get_max_possible_connections(
        )
        self.num_neurons = self._get_num_neurons(
        )
        self.min_connectivity = self._get_min_connectivity()
        self.min_connectivity_input_output = self._get_min_connectivity_input_output()
        self.num_epochs = self._get_num_epochs()
        self.task = self._get_task()
        self.optimizer = self._get_optimizer()
        self.model_cls = self._get_model_cls()
        self.structural_features_vec_length_with_motifs = self._get_structural_features_vec_length_with_motifs()
        self.structural_features_full_name_vec_with_motifs = self._get_structural_features_full_name_vec_with_motifs()

        self.max_in_connections_per_layers = self._get_max_in_connections_per_layers()
        self.max_out_connections_per_layers = self._get_max_out_connections_per_layers()
        fully_connected_graph = self._get_fully_connected_graph()
        self.max_num_paths_to_output_per_input_neuron = self._get_max_num_paths_to_output_per_input_neuron(
            fully_connected_graph=fully_connected_graph,
        )
        self.max_num_involved_neurons_in_paths_per_input_neuron = self._get_max_num_involved_neurons_in_paths_per_input_neuron()
        self.ff_entropy = self._get_ff_entropy(
            fully_connected_graph=fully_connected_graph,
        )
        self.ff_normed_entropy = self._get_ff_normed_entropy()
        self.max_motif_count = self._get_max_motif_count(
            fully_connected_graph=fully_connected_graph,
        )

    @classmethod
    def _get_start_dimensions(cls):
        return cls.start_dimensions

    def _get_max_possible_connections(self) -> int:
        return sum(self.start_dimensions[i] * self.start_dimensions[i + 1] for i in range(self.num_layers))

    def _get_num_neurons(self) -> int:
        return sum(self.start_dimensions)

    def _get_min_connectivity(self) -> float:
        return round(self.num_neurons / self.max_possible_connections, 2)

    def _get_min_connectivity_input_output(self) -> float:
        return round((self.start_dimensions[0] + self.start_dimensions[-1]) / self.max_possible_connections, 2)

    def _get_num_epochs(self) -> int:
        pass

    @classmethod
    def _get_task(self) -> Optional[Task]:
        pass

    @classmethod
    def _get_optimizer(cls) -> Optional[torch.optim.Optimizer]:
        pass

    @classmethod
    def _get_structural_features_vec_length_with_motifs(cls) -> Optional[int]:
        if cls.structural_features_vec_length is None:
            return
        return cls.structural_features_vec_length + len(motifs_full_name)

    @classmethod
    def _get_structural_features_full_name_vec_with_motifs(cls) -> Optional[List[str]]:
        if cls.structural_features_full_name_vec is None:
            return
        return cls.structural_features_full_name_vec + motifs_full_name

    @property
    def teach_arch_base_path(cls) -> str:
        if cls.task_base_folder_name is None:
            cls.task_base_folder_name = cls.task_global_name
        return f"{cls.base_path}/{cls.task_base_folder_name}/{cls.task_version_name}"

    @property
    def output_folder(cls) -> str:
        return f"{cls.teach_arch_base_path}/teach_archs_results"

    @property
    def pkls_folder(cls) -> str:
        return f"{cls.teach_arch_base_path}/teach_archs_models"

    def _get_max_in_connections_per_layers(self) -> List[int]:
        return [0] + self.start_dimensions[:-1]

    def _get_max_out_connections_per_layers(self) -> List[int]:
        return self.start_dimensions[1:] + [0]

    def _get_fully_connected_graph(self):
        fully_connected_graph = np.zeros((self.num_neurons, self.num_neurons))
        for layer in range(len(self.start_dimensions) - 1):
            start_ind = sum(self.start_dimensions[:layer])
            ff_block_min_row = start_ind
            ff_block_max_row = start_ind + self.start_dimensions[layer]
            ff_block_min_column = start_ind + self.start_dimensions[layer]
            ff_block_max_column = start_ind + self.start_dimensions[layer] + self.start_dimensions[layer + 1]
            fully_connected_graph[ff_block_min_row:ff_block_max_row, ff_block_min_column:ff_block_max_column] = 1
        return fully_connected_graph

    def _get_max_num_paths_to_output_per_input_neuron(self, fully_connected_graph) -> List[int]:
        return np.sum(
            np.linalg.matrix_power(fully_connected_graph, self.num_layers), axis=1)[
               :self.input_dim]

    def _get_max_num_involved_neurons_in_paths_per_input_neuron(self) -> List[int]:
        return [self.num_neurons - self.input_dim] * self.input_dim

    @classmethod
    def _get_ff_entropy(cls, fully_connected_graph) -> float:
        in_degrees = sum(fully_connected_graph[:, ])
        prob_list = [
            neuron_in_degree / sum(in_degrees)
            for neuron_in_degree in in_degrees
            if neuron_in_degree != 0
        ]
        return -(prob_list * np.log(np.abs(prob_list))).sum()

    def _get_ff_normed_entropy(self) -> float:
        return self.ff_entropy / np.log(self.num_neurons)

    @classmethod
    def _get_max_motif_count(cls, fully_connected_graph) -> List[int]:
        return [
            sum(comb(np.sum(fully_connected_graph, axis=1), 2)),
            np.matmul(np.sum(fully_connected_graph, axis=0), np.sum(fully_connected_graph, axis=1)),
            sum(comb(np.sum(fully_connected_graph, axis=0), 2)),
        ]

    def _get_model_cls(self) -> Optional[NeuralNetModel]:
        nn_model_mapping = {
            1: FeedforwardNeuralNetModelOneLayer,
            2: FeedforwardNeuralNetModelTwoLayers,
            3: FeedforwardNeuralNetModelThreeLayers,
            4: FeedforwardNeuralNetModelFourLayers,
            5: FeedforwardNeuralNetModelFiveLayers,
            6: FeedforwardNeuralNetModelSixLayers,
        }
        return nn_model_mapping[self.num_layers]

    def _get_middle_dim_permutations(self) -> Set[List[int]]:
        all_possible_middle_dims = []
        for possible_mid_dim in self.possible_middle_dims_options:
            all_possible_middle_dims += list(permutations(possible_mid_dim))
        return set(all_possible_middle_dims)
