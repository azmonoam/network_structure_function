import random
from typing import List, Callable

import numpy as np
from deap.tools.emo import selNSGA2

from base_ecosystem import BaseEcosystem
from fitness_calc import FitnessCalc, PerformanceAndConnectionCost, OnlyPerformance
from neuroevolution import Neuroevolution
from organism import Organism
from reproduction import Reproduction


class Ecosystem(BaseEcosystem):
    def __init__(
            self,
            neuroevolution: Neuroevolution,
    ):
        population_size = neuroevolution.population_size
        potential_parents_percent = neuroevolution.potential_parents_percent
        super().__init__(population_size, potential_parents_percent)
        self.population_size = neuroevolution.population_size
        self.connection_cost_method = neuroevolution.connection_cost_method
        self.inputs = neuroevolution.inputs
        self.optimization_method = neuroevolution.optimization_method
        self.optimize_by_connection_in_prob = neuroevolution.optimize_by_connection_in_prob
        self.potential_parents_percent = neuroevolution.potential_parents_percent

        self.neuroevolution = neuroevolution

    def generation(
            self,
            population: List[Organism],
            labels: np.ndarray,
            generation: int,
    ) -> List[Organism]:
        fitness_calc_method = OnlyPerformance
        if self.optimization_method == 'performance_and_connection_cost':
            if random.uniform(0, 1) < self.optimize_by_connection_in_prob:
                fitness_calc_method = PerformanceAndConnectionCost
        self._add_fitness_to_population(
            population=population,
            labels=labels,
            fitness_calc_method=fitness_calc_method,
        )
        potential_parents = self._choose_parents_one_best_one_random(
            population=population,
        )
        offsprings = self._get_offsprings(
            potential_parents=potential_parents,
            generation=generation,
        )

        self._add_fitness_to_population(
            population=offsprings,
            labels=labels,
            fitness_calc_method=fitness_calc_method,
        )

        potential_next_gen = population + offsprings
        random.shuffle(potential_next_gen)
        next_gen = self._get_best_organisms(
            required_number=self.population_size,
            population=potential_next_gen,
        )

        return next_gen

    def _get_best_organisms(
            self,
            required_number: int,
            population: List[Organism]
    ) -> List[Organism]:
        best_organisms = selNSGA2(population, required_number, nd='standard')
        if self.optimization_method == 'performance_and_connection_cost':
            best_organisms.sort(key=lambda x: (x.performance, -x.connection_cost), reverse=True)
        else:
            best_organisms.sort(key=lambda x: x.fitness.values, reverse=True)
        return best_organisms

    def _get_offsprings(
            self,
            potential_parents: List[List[Organism]],
            generation: int,
    ) -> List[Organism]:
        offsprings = []
        for parent_a, parent_b in potential_parents:
            reproduction = Reproduction(
                first_parent=parent_a,
                second_parent=parent_b,
                neuroevolution=self.neuroevolution,
            )
            offspring = reproduction.mate()
            offspring.generation = generation
            offspring.predictions = offspring.predict(
                input_array=self.inputs,
            )
            offsprings.append(offspring)

        return offsprings

    def _add_fitness_to_population(
            self,
            population: List[Organism],
            labels: np.ndarray,
            fitness_calc_method: Callable,
    ):
        for organism in population:
            fitness_calc = FitnessCalc(
                connection_cost_method=self.connection_cost_method,
                organism=organism,
                labels=labels,
                fitness_calc_method=fitness_calc_method,
            )
            fitness_calc.calc_and_add_fitness()
