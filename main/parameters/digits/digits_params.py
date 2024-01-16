from typing import Callable, Dict, List, Optional

import torch

from logical_gates import LogicalGates
from parameters.base_params import BaseParams
from tasks import DigitsTask


class DigitsParams(BaseParams):
    task_global_name: str = 'digits'

    communities_input_symmetry: int = None
    input_dim: int = 64
    output_dim: int = 10

    mutation_probabilities: Dict[str, float] = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    learning_rate: float = 0.005
    activate: str = 'relu'
    flatten: bool = False

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

    def _get_task(self) -> DigitsTask:
        return DigitsTask(
            input_dim=self.input_dim,
            rule=None
        )

    @classmethod
    def _get_optimizer(cls):
        return torch.optim.Adam
