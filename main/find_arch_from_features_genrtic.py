import random
from typing import Set, Optional, Tuple, List

import numpy as np
from tqdm import tqdm

from find_feature_spectrum.find_feature_dist import FindFeaturesDist
from generation_ecosysten import GenerationEcosystem
from new_organism import Organism
from parameters.base_params import BaseParams
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from utils.set_up_population_utils import get_organism_by_connectivity_ratio


class FindArchGenetic:
    def __init__(
            self,
            generations,
            task_params: BaseParams,
            selected_feature_names: List[str],
            population_size,
            potential_parents_percent,
            connectivity_ratio: Optional[float] = None,
            target_feature_values: Optional[List[float]] = None,
            find_feature_dist: Optional[FindFeaturesDist] = None,
            use_distance_fitness: bool = True,
            dist_stds: Optional[np.ndarray] = None,
            connectivity_in_perc: Optional[bool] = False,
    ):
        self.generations = generations
        self.task_params = task_params
        self.connectivity_ratio = connectivity_ratio
        self.population_size = population_size
        self.potential_parents_percent = potential_parents_percent
        self.selected_feature_names = selected_feature_names
        self.find_feature_dist = find_feature_dist
        self.use_distance_fitness = use_distance_fitness
        self.dist_stds = dist_stds

        self.ecosystem = GenerationEcosystem(
            task_params=self.task_params,
            population_size=self.population_size,
            potential_parents_percent=self.potential_parents_percent,
            features_names=self.selected_feature_names,
            target_feature_values=target_feature_values,
            find_feature_dist=self.find_feature_dist,
            use_distance_fitness=self.use_distance_fitness,
            dist_stds=self.dist_stds,
            connectivity_in_perc=connectivity_in_perc,
        )

    def _get_population(self) -> List[Organism]:
        connectivities = [
            self.connectivity_ratio
            if self.connectivity_ratio is not None
            else round(random.uniform(self.task_params.min_connectivity, 1.0), 2)
            for _ in range(self.population_size)
        ]
        return [
            get_organism_by_connectivity_ratio(
                task_params=self.task_params,
                connectivity_ratio=connectivity_ratio,
            )
            for connectivity_ratio in connectivities
        ]

    def _get_x_archs_with_features_genetic(
            self,
            num_orgs_to_return: int,
            mse_early_stopping_criteria_factor: float = 0.005,
            distance_early_stopping_criteria_num_sigmas: float = 0.1,
            duplicates_orgs_features_values: Optional[Set[Tuple[float]]] = None,
            errors: Optional[np.ndarray] = None,
            test_saved_orgs_error: bool = False,
    ) -> Set[Organism]:
        population = self._get_population()
        early_stopping_criteria, stopping_function = self._get_stopping_criteria_and_function(
            distance_early_stopping_criteria_num_sigmas=distance_early_stopping_criteria_num_sigmas,
            mse_early_stopping_criteria_factor=mse_early_stopping_criteria_factor,
            errors=errors
        )
        orgs_to_save = set()
        saved_orgs_feature_values = []
        if duplicates_orgs_features_values is None:
            duplicates_orgs_features_values = set()
        best_organisms = []
        counts = []
        for i in tqdm(range(1, self.generations + 1), "generation"):
            population = self.ecosystem.generation(
                population=population,
                generation=i,
            )
            this_generation_best = population[0]
            best_organisms.append(this_generation_best)
            print(f'feature values: {this_generation_best.features_values}')
            print(f'fitness values: {this_generation_best.fitness.values}')
            print(f'sum fitness values: {sum(this_generation_best.fitness.values)}')

            c = 0
            if stopping_function(
                    organism=this_generation_best,
                    early_stopping_criteria=early_stopping_criteria,
            ):
                for j, org in enumerate(population):
                    if stopping_function(
                            organism=org,
                            early_stopping_criteria=early_stopping_criteria,
                    ):
                        if org in orgs_to_save:
                            continue
                        structural_features_calculator = CalcStructuralFeatures(
                            organism=org,
                        )
                        try:
                            org = structural_features_calculator.calc_structural_features()
                        except:
                            continue
                        duplicates_values = tuple(org.structural_features.get_class_values())
                        if duplicates_values in duplicates_orgs_features_values:
                            continue
                        duplicates_orgs_features_values.add(duplicates_values)
                        orgs_to_save.add(org)
                        saved_orgs_feature_values.append(org.features_values)
                        c += 1
                print(f'saved {c} orgs')
                counts.append(c)
                if test_saved_orgs_error:
                    saved_orgs_error = [
                        abs(mean_obs_stat - target) / std

                        for mean_obs_stat, target, std
                        in zip(np.mean(saved_orgs_feature_values, axis=0), self.ecosystem.target_feature_values, errors)
                    ]
                    print(f'mean saved orgs error: {saved_orgs_error}')
                    if sum(1 for error in saved_orgs_error if error < 1) == len(saved_orgs_error):
                        break
                if len(orgs_to_save) > num_orgs_to_return or sum(counts[-20:]) == 0:
                    break
        return orgs_to_save

    def _get_stopping_criteria_and_function(
            self,
            distance_early_stopping_criteria_num_sigmas: float,
            mse_early_stopping_criteria_factor: float,
            errors: Optional[np.ndarray],
    ):
        if self.use_distance_fitness:
            early_stopping_criteria = distance_early_stopping_criteria_num_sigmas
            stopping_function = self._distance_stopping_func
        else:
            if self.dist_stds is not None:
                early_stopping_criteria = [
                    1
                    for _ in self.ecosystem.target_feature_values
                ]
            elif errors is not None:
                early_stopping_criteria = errors.tolist()
            else:
                early_stopping_criteria = [
                    mse_early_stopping_criteria_factor
                    for _ in self.ecosystem.target_feature_values
                ]
            stopping_function = self._mse_stopping_func
        return early_stopping_criteria, stopping_function

    @staticmethod
    def _distance_stopping_func(
            organism: Organism,
            early_stopping_criteria,
    ):
        return organism.fitness.values[0] < early_stopping_criteria

    @staticmethod
    def _mse_stopping_func(
            organism: Organism,
            early_stopping_criteria,
    ):
        is_within_stopping_criteria = [
            mse < limit
            for mse, limit in zip(organism.fitness.values, early_stopping_criteria)
        ]
        return all(is_within_stopping_criteria)
