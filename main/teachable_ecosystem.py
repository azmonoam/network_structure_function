import multiprocessing
import random
from typing import List, Optional

import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed

from ecosystem import Ecosystem
from neuroevolution import Neuroevolution
from nn_models.nn_model_four_layers import FeedforwardNeuralNetModelFourLayers
from organism import Organism
from reproduction import Reproduction
from tasks import Task, RetinaTask
from teach_arch import TeachArch
from utils.main_utils import get_all_unused_nodes, get_number_of_connections


class TeachableEcosystem(Ecosystem):
    def __init__(
            self,
            neuroevolution: Neuroevolution,
            learning_rate: Optional[float],
            num_epochs: Optional[int],
            reinitiate: Optional[bool],
            model_cls: nn.Module = FeedforwardNeuralNetModelFourLayers,
            task: Task = RetinaTask,
            activate: str = 'tanh',
    ):
        super().__init__(neuroevolution)
        self.teach_arch = TeachArch(
            rule=neuroevolution.rules,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            reinitiate=reinitiate,
            input_dim=neuroevolution.input_dim,
            model_cls=model_cls,
            task=task,
            activate=activate,
        )
        self.num_cores = multiprocessing.cpu_count()
        print(f"-- using {self.num_cores} cores --")

    def generation(
            self,
            population: List[Organism],
            labels: np.ndarray,
            generation: int,
    ) -> List[Organism]:
        population = self._add_performance_to_population_parallel_wrapper(
            population=population,
        )
        potential_parents = self._choose_parents_one_best_one_random(
            population=population,
        )
        offsprings = self._get_offsprings(
            potential_parents=potential_parents,
            generation=generation,
        )
        offsprings = self._add_performance_to_population_parallel_wrapper(
            population=offsprings,
        )
        potential_next_gen = population + offsprings
        random.shuffle(potential_next_gen)
        next_gen = self._get_best_organisms(
            required_number=self.population_size,
            population=potential_next_gen,
        )
        return next_gen

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
            offsprings.append(offspring)
        return offsprings

    def _add_performance_to_population(
            self,
            organism: Organism,
    ):
        unused_nodes = get_all_unused_nodes(
            output_layer=len(organism.dimensions) - 1,
            input_layer=0,
            network=organism.network,
        )
        if len(set(organism.network.nodes) - unused_nodes) <= organism.dimensions[-1]:
            performance = 0
            get_all_unused_nodes(
                output_layer=len(organism.dimensions) - 1,
                input_layer=0,
                network=organism.network,
            )
        else:
            performance = self.teach_arch.back_prop_teach_arch(
                organism=organism,
            )
        organism.performance = performance
        organism.connection_cost = get_number_of_connections(
            organism=organism,
        )
        organism.fitness = (organism.performance,)
        return organism

    def _add_performance_to_population_parallel_wrapper(
            self,
            population: List[Organism],
    ) -> List[Organism]:
        return Parallel(n_jobs=self.num_cores)(
            delayed(self._add_performance_to_population)(organism)
            for organism in population
        )

    def _get_best_organisms(
            self,
            required_number: int,
            population: List[Organism]
    ) -> List[Organism]:
        population.sort(key=lambda x: x.performance, reverse=True)
        return population[:required_number]
