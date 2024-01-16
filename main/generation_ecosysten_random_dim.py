from typing import List, Optional, Dict

import numpy as np

from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from generation_ecosysten_base import GenerationEcosystemBase
from new_organism import Organism
from new_reproduction_random_dim import NReproductionRandomDim


class GenerationEcosystemRandomDim(GenerationEcosystemBase):
    def __init__(
            self,
            population_size,
            potential_parents_percent,
            features_names: List[str],
            mutation_probabilities: Dict[str, float],
            target_feature_values: Optional[List[float]] = None,
            find_feature_dist: Optional[FindFeaturesDistByPerformance] = None,
            use_distance_fitness: bool = True,
            dist_stds: Optional[np.ndarray] = None,
            connectivity_in_perc: Optional[bool] = False,
    ):
        super().__init__(population_size, potential_parents_percent, features_names, target_feature_values,
                         find_feature_dist, use_distance_fitness, dist_stds, connectivity_in_perc)
        self.mutation_probabilities = mutation_probabilities

    def _get_offsprings(
            self,
            potential_parents: List[List[Organism]],
            generation: int,
    ) -> List[Organism]:
        offsprings = []
        for parent_a, parent_b in potential_parents:
            reproduction = NReproductionRandomDim(
                first_parent=parent_a,
                second_parent=parent_b,
                mutation_probabilities=self.mutation_probabilities,
            )
            offspring = reproduction.mate()
            offspring.generation = generation
            offsprings.append(offspring)
        return offsprings
