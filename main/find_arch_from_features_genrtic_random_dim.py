import random
from typing import Optional, List, Union

import numpy as np

from find_arch_from_features_genrtic import FindArchGenetic
from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance
from generation_ecosysten_random_dim import GenerationEcosystemRandomDim
from new_organism import Organism
from parameters.base_params import BaseParams
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.xor.xor_by_dim import XoraByDim
from utils.set_up_population_utils import get_organism_by_connectivity_ratio


class FindArchGeneticRandomDim(FindArchGenetic):
    def __init__(
            self,
            generations,
            task_params_example: BaseParams,
            selected_feature_names: List[str],
            population_size,
            potential_parents_percent,
            task_class:  Union[XoraByDim, RetinaByDim, DigitsByDim],
            connectivity_ratio: Optional[float] = None,
            target_feature_values: Optional[List[float]] = None,
            find_feature_dist: Optional[FindFeaturesDistByPerformance] = None,
            use_distance_fitness: bool = True,
            dist_stds: Optional[np.ndarray] = None,
    ):
        super().__init__(generations, task_params_example, selected_feature_names, population_size, potential_parents_percent,
                         connectivity_ratio, target_feature_values, find_feature_dist, use_distance_fitness, dist_stds)

        self.task_class = task_class

        self.ecosystem = GenerationEcosystemRandomDim(
            population_size=self.population_size,
            potential_parents_percent=self.potential_parents_percent,
            features_names=self.selected_feature_names,
            target_feature_values=target_feature_values,
            find_feature_dist=self.find_feature_dist,
            use_distance_fitness=self.use_distance_fitness,
            dist_stds=self.dist_stds,
            mutation_probabilities=task_class.mutation_probabilities
        )

    def _get_population(
            self,
    ) -> List[Organism]:
        middle_dim_permutations = self.task_params._get_middle_dim_permutations()
        possible_dims = [
            [self.task_params.start_dimensions[0]] + list(middle_dim) + [self.task_params.start_dimensions[-1]]
            for middle_dim in middle_dim_permutations
        ]
        task_params_list = [
            self.task_class(
                start_dimensions=random.choice(possible_dims),
                num_layers=self.task_params.num_layers,
                by_epochs=False,
                task_base_folder_name=self.task_params.task_base_folder_name,
                rule=self.task_params.rule,
            )
            for _ in range(self.population_size)
        ]
        connectivities = [
            self.connectivity_ratio
            if self.connectivity_ratio is not None
            else round(random.uniform(task_params_list[i].min_connectivity, 1.0), 2)
            for i in range(self.population_size)
        ]
        return [
            get_organism_by_connectivity_ratio(
                task_params=task_params,
                connectivity_ratio=connectivity_ratio,
            )
            for task_params, connectivity_ratio in zip(task_params_list, connectivities)
        ]
