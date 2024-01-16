from typing import Callable, Dict, List, Optional
import torch
import numpy as np

from logical_gates import LogicalGates
from parameters.base_params import BaseParams
from tasks import RetinaTask


class RetinaParams(BaseParams):
    task_global_name: str = 'retina'

    communities_input_symmetry: int = 2
    input_dim: int = 6
    output_dim: int = 2

    mutation_probabilities: Dict[str, float] = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    learning_rate: float = 0.1
    activate: str = 'tanh'
    flatten: bool = False
    possible_middle_dims_options = [
        [5, 2],
        [3, 4],
    ]

    def __init__(
            self,
            start_dimensions: List[int],
            rule: Optional[Callable] = None,
    ):
        if rule is not None:
            self.rule = rule
        else:
            self.rule: Callable = LogicalGates.AND
        super().__init__(start_dimensions)

    def _get_task(self) -> RetinaTask:
        return RetinaTask(
            input_dim=self.input_dim,
            rule=self.rule,
        )

    @classmethod
    def _get_optimizer(cls):
        return torch.optim.SGD

    def _get_num_epochs(self) -> int:
        return int(np.floor((self.max_possible_connections * 25) / 100) * 100)
