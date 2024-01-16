import random
from typing import List, Optional

import numpy as np
from deap.tools.emo import selNSGA2

from base_ecosystem import BaseEcosystem
from ergm_model.data_halper import get_list_of_functions_from_features_names
from ergm_model.nx_methods import NxMethods
from find_feature_spectrum.find_feature_dist import FindFeaturesDist
from fitness_calc import FitnessMin
from new_organism import Organism


class GenerationEcosystemBase(BaseEcosystem):
    def __init__(
            self,
            population_size,
            potential_parents_percent,
            features_names: List[str],
            target_feature_values: Optional[List[float]] = None,
            find_feature_dist: Optional[FindFeaturesDist] = None,
            use_distance_fitness: bool = True,
            dist_stds: Optional[np.ndarray] = None,
            connectivity_in_perc: Optional[bool] = False,
    ):
        super().__init__(population_size, potential_parents_percent)
        self.methods = NxMethods(
            connectivity_in_perc=connectivity_in_perc
        )
        self.features_names = features_names
        self.features = get_list_of_functions_from_features_names(
            method_class=self.methods,
            features_names_list=features_names,
        )
        self.use_distance_fitness = use_distance_fitness
        self.fitness_calc_method = FitnessMin
        if not self.use_distance_fitness:
            self.fitness_calc_method.weights = tuple(
                -1.0
                for _ in range(len(features_names))
            )
        self._init_target_features(
            target_feature_values=target_feature_values,
            find_feature_dist=find_feature_dist,
        )
        self.dist_stds = dist_stds
        self._test_inputs()

    def _init_target_features(
            self,
            target_feature_values: Optional[List[float]],
            find_feature_dist: Optional[FindFeaturesDist],
    ):
        if target_feature_values:
            self.target_feature_values = target_feature_values
        elif find_feature_dist:
            self.target_feature_values = find_feature_dist.target_mean_features
        else:
            raise ValueError("Either 'find_feature_dist' or 'target_feature_values' need to be set")
        if self.use_distance_fitness:
            self.calc_distance_from_target = find_feature_dist.calc_distance_from_mean

    def _test_inputs(self):
        if not self.use_distance_fitness and self.dist_stds is None:
            raise ValueError("Either 'use_distance_fitness' or 'dist_stds' need to be set")

    def generation(
            self,
            population: List[Organism],
            generation: int,
    ) -> List[Organism]:
        self._add_fitness_to_population(
            population=population,
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
        )

        potential_next_gen = population + offsprings
        random.shuffle(potential_next_gen)
        next_gen = self._get_best_organisms(
            required_number=self.population_size,
            population=potential_next_gen,
        )
        return next_gen

    def _add_fitness_to_population(
            self,
            population: List[Organism],
    ):
        for organism in population:
            organism.fitness = self.fitness_calc_method()
            features_values = [
                f(organism, f_name)
                for f, f_name in zip(self.features, self.features_names)
            ]
            organism.features_values = features_values
            if self.use_distance_fitness:
                distance = self.calc_distance_from_target(
                    sample=np.array(features_values)
                )
                organism.fitness.values = (distance,)
            else:
                organism.fitness.values = tuple(
                    [
                        abs(mean_obs_stat - target) / std
                        for mean_obs_stat, target, std
                        in zip(features_values, self.target_feature_values, self.dist_stds)
                    ]
                )

    def _get_best_organisms(
            self,
            required_number: int,
            population: List[Organism]
    ) -> List[Organism]:
        best_organisms = selNSGA2(population, required_number, nd='standard')
        best_organisms.sort(key=lambda x: sum(x.fitness.values), reverse=False)
        return best_organisms
