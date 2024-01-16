from typing import List, Dict, Callable, Optional, Union

import numpy as np

from tasks import Task


class Neuroevolution:

    def __init__(
            self,
            rules: Union[List[Callable], Callable],
            num_layers: int,
            input_dim: int,
            allowed_weights_values: List[int],
            allowed_bias_values: Optional[List[int]],
            population_size: Optional[int],
            generations: Optional[int],
            mutation_probabilities: Dict[str, float],
            connection_cost_method: Optional[str],
            task: Task,
            optimization_method: Optional[str] = None,
            potential_parents_percent: Optional[int] = None,
            optimize_by_connection_in_prob: Optional[float] = None,
    ):
        self.rules = rules
        self.num_layers = num_layers
        self.allowed_weights_values = allowed_weights_values
        self.allowed_bias_values = allowed_bias_values
        self.population_size = population_size
        self.generations = generations
        self.connection_cost_method = connection_cost_method
        self.potential_parents_percent = potential_parents_percent
        self.mutation_probabilities = mutation_probabilities
        self.optimization_method = optimization_method
        self.optimize_by_connection_in_prob = optimize_by_connection_in_prob
        self.input_dim = input_dim
        self.task = task
        data_fetcher = self.task.data_fetcher()
        inputs = data_fetcher.get_inputs()
        self.labels = np.array(
            data_fetcher.get_labels(
                inputs=inputs,
                rule=self.rules,
            ),
        )
        self.inputs = np.array(inputs)
