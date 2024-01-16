import random
from typing import List

import numpy as np

from organism import Organism


class BaseEcosystem:
    def __init__(
            self,
            population_size: int,
            potential_parents_percent: float,
    ):
        self.population_size = population_size
        self.potential_parents_percent = potential_parents_percent

    def _get_best_organisms(
            self,
            required_number: int,
            population: List[Organism]
    ) -> List[Organism]:
        pass

    def _choose_parents(
            self,
            population: List[Organism],
    ) -> List[List[Organism]]:
        number_of_parents = int(self.population_size * (self.potential_parents_percent / 100))
        highest_fitness_organisms = self._get_best_organisms(
            required_number=number_of_parents,
            population=population,
        )
        return [
            random.sample(list(highest_fitness_organisms), 2)
            for _ in range(self.population_size)
        ]

    def _choose_parents_one_best_one_random(
            self,
            population: List[Organism],
    ) -> List[List[Organism]]:
        parents = []
        number_of_parents = int(self.population_size * (self.potential_parents_percent / 100))
        highest_fitness_organisms = self._get_best_organisms(
            required_number=number_of_parents,
            population=population,
        )
        while len(parents) < self.population_size:
            for first_parent in highest_fitness_organisms:
                second_parent = population[int(np.random.uniform(0, self.population_size - 1))]
                parents.append([first_parent, second_parent])
                if len(parents) == self.population_size:
                    return parents

    def _get_offsprings(
            self,
            potential_parents: List[List[Organism]],
            generation: int,
    ) -> List[Organism]:
        pass
