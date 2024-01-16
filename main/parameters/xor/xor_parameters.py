from typing import Callable, Dict
import torch
import numpy as np

from logical_gates import LogicalGates
from parameters.base_params import BaseParams
from tasks import XorsTask


class XorParams(BaseParams):
    task_global_name: str = 'xor'

    rule: Callable = LogicalGates.XOR
    communities_input_symmetry: int = 3
    input_dim: int = 6

    mutation_probabilities: Dict[str, float] = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    learning_rate: float = 0.001
    activate: str = 'relu'
    flatten: bool = False
    possible_middle_dims_options = [
        [6, 6, 2],
        [6, 5, 3],
        [6, 4, 4],
        [5, 5, 4],
    ]

    @classmethod
    def _get_task(cls) -> XorsTask:
        return XorsTask(
            input_dim=cls.input_dim,
            rule=cls.rule,
        )

    @classmethod
    def _get_optimizer(cls):
        return torch.optim.Adam

    def _get_num_epochs(self) -> int:
        return int(np.floor((self.max_possible_connections * 70) / 100) * 100)
