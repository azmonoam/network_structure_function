from typing import List, Optional

import numpy as np

from find_feature_spectrum.find_feature_dist import FindFeaturesDist
from generation_ecosysten_base import GenerationEcosystemBase
from new_organism import Organism
from new_reproduction import NReproduction
from parameters.base_params import BaseParams


class GenerationEcosystem(GenerationEcosystemBase):
    def __init__(
            self,
            population_size,
            potential_parents_percent,
            task_params: BaseParams,
            features_names: List[str],
            target_feature_values: Optional[List[float]] = None,
            find_feature_dist: Optional[FindFeaturesDist] = None,
            use_distance_fitness: bool = True,
            dist_stds: Optional[np.ndarray] = None,
            connectivity_in_perc: Optional[bool] = False,
    ):
        super().__init__(population_size, potential_parents_percent, features_names, target_feature_values,
                         find_feature_dist, use_distance_fitness, dist_stds, connectivity_in_perc)
        self.task_params = task_params
        self.mutation_probabilities = task_params.mutation_probabilities

    def _get_offsprings(
            self,
            potential_parents: List[List[Organism]],
            generation: int,
    ) -> List[Organism]:
        offsprings = []
        for parent_a, parent_b in potential_parents:
            reproduction = NReproduction(
                first_parent=parent_a,
                second_parent=parent_b,
                mutation_probabilities=self.mutation_probabilities,
            )
            offspring = reproduction.mate()
            offspring.generation = generation
            offsprings.append(offspring)
        return offsprings
