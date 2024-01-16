from typing import Callable

import deap.base as base
import numpy as np

from organism import Organism
from utils.main_utils import get_number_of_connections


class FitnessMin(base.Fitness):
    weights = (-1.0,)


class FitnessMinMax(base.Fitness):
    weights = (1.0, -1.0)


class PerformanceAndConnectionCost(base.Fitness):
    weights = (1.0, -1.0)


class OnlyPerformance(base.Fitness):
    weights = (1.0,)


class FitnessMin2Features(base.Fitness):
    weights = (-1.0, -1.0,)


class FitnessMin4Features(base.Fitness):
    weights = (-1.0, -1.0, -1.0, -1.0,)


class FitnessMin5Features(base.Fitness):
    weights = (-1.0, -1.0, -1.0, -1.0, -1.0,)


class FitnessMin20Features(base.Fitness):
    weights = (
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    )


class FitnessMin10Features(base.Fitness):
    weights = (
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    )


class FitnessCalc:
    def __init__(
            self,
            organism: Organism,
            connection_cost_method: str,
            labels: np.ndarray,
            fitness_calc_method: Callable,
    ):
        self.organism = organism
        self.connection_cost_method = connection_cost_method
        self.labels = labels
        self.fitness_calc_method = fitness_calc_method

    def calc_and_add_fitness(self):
        self.add_performance_cost()
        self.add_connection_cost()
        self.organism.fitness = self.fitness_calc_method()
        if self.fitness_calc_method == OnlyPerformance:
            self.organism.fitness.values = (self.organism.performance,)
        else:
            self.organism.fitness.values = (self.organism.performance, self.organism.connection_cost)

    def add_performance_cost(self):
        self.organism.performance = np.mean(self.labels.flatten() == np.argmax(self.organism.predictions, axis=1))

    def add_connection_cost(self):
        if self.connection_cost_method == 'num_connections':
            connection_cost = self._calc_num_connection()
        elif self.connection_cost_method == 'connection_length':
            connection_cost = self._calc_sqr_of_length()
        else:
            connection_cost = 0
        self.organism.connection_cost = connection_cost

    def _calc_num_connection(self) -> int:
        return get_number_of_connections(self.organism)

    def _calc_sqr_of_length(self):
        return 0
