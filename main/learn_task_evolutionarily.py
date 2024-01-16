from typing import Optional, List, Callable

import torch.nn as nn
from tqdm import tqdm

from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from ecosystem import Ecosystem
from fitness_calc import FitnessCalc, OnlyPerformance
from neuroevolution import Neuroevolution
from organism import Organism
from plotting.plotting import draw_network_and_performance
from teach_arch_multiple_times import TeachArchMultiTime
from utils.main_utils import get_performance_list, print_log, set_up_experiment_folders, export_results_to_csv, save_to_pkl


class LearnTaskEvolutionarily:
    def __init__(
            self,
            neuroevolution: Neuroevolution,
            base_path: Optional[str],
            experiment_name: Optional[str],
            save_every: int = 100,
    ):
        self.neuroevolution = neuroevolution
        self.base_path = base_path
        self.experiment_name = experiment_name
        self.best_folder, self.current_experiment_folder = set_up_experiment_folders(
            experiment_name=experiment_name,
            base_path=self.base_path,
        )
        self.save_every = save_every

    def _teach_good_arch(
            self,
            good_archs: List[Organism],
            good_archs_names: List[str],
            num_exp_per_arch: int,
            learning_rate: float,
            num_epochs: int,
            teach_arch_folder_name: str,
            model_cls: Optional[nn.Module],
            activate: Optional[str],
    ):
        for arch, exp_name in zip(good_archs, good_archs_names):
            output_path = f"{self.base_path}/{teach_arch_folder_name}/{exp_name}_teach.csv"
            teach_arch = TeachArchMultiTime(
                input_dim=self.neuroevolution.input_dim,
                output_path=output_path,
                exp_name=exp_name,
                rule=self.neuroevolution.rules,
                model_cls=model_cls,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                num_exp_per_arch=num_exp_per_arch,
                task=self.neuroevolution.task,
                activate=activate,
            )
            teach_arch.teach_arch_many_times_parallel(
                organism=arch,
            )

    def _add_eval_to_single_organism(
            self,
            organism: Organism,
            fitness_calc_method: Callable,
    ):
        calc_structural_features = CalcStructuralFeatures(
            organism=organism,
        )
        calc_structural_features.add_modularity()
        fitness_calc = FitnessCalc(
            connection_cost_method=self.neuroevolution.connection_cost_method,
            organism=organism,
            labels=self.neuroevolution.labels,
            fitness_calc_method=fitness_calc_method,
        )
        fitness_calc.add_connection_cost()
        fitness_calc.add_performance_cost()

    def learn_evolutionarily(
            self,
            population: List[Organism],
    ):
        self._add_eval_to_single_organism(
            organism=population[0],
            fitness_calc_method=OnlyPerformance,
        )
        save_to_pkl(
            best_folder=self.best_folder,
            this_generation_best=population[0],
            generation=0,
        )
        best_organisms = [population[0]]
        ecosystem = Ecosystem(
            neuroevolution=self.neuroevolution,
        )
        for i in tqdm(range(1, self.neuroevolution.generations + 1), "generation"):
            population = ecosystem.generation(
                population=population,
                labels=self.neuroevolution.labels,
                generation=i,
            )
            this_generation_best = population[0]
            calc_structural_features = CalcStructuralFeatures(
                organism=this_generation_best,
            )
            calc_structural_features.add_modularity()
            best_organisms.append(this_generation_best)
            if i % self.save_every == 0:
                save_to_pkl(
                    best_folder=self.best_folder,
                    this_generation_best=this_generation_best,
                    generation=i,
                )
                print_log(
                    neuroevolution=self.neuroevolution,
                    this_generation_best=this_generation_best,
                    generation=i,
                )
        performances = get_performance_list(
            organisms=best_organisms,
        )
        export_results_to_csv(
            organisms=best_organisms,
            performances=performances,
            experiment_folder=self.current_experiment_folder,
        )
        draw_network_and_performance(
            best_organisms=best_organisms,
            performances=performances,
            folder=self.current_experiment_folder,
            optimization_method=self.neuroevolution.optimization_method,
            plot_every=10
        )

    def learn_evolutionarily_for_teachability_with_teaching(
            self,
            population: List[Organism],
            model_cls: Optional[nn.Module],
            activate: Optional[str],
            num_exp_per_arch: Optional[int],
            learning_rate: Optional[float],
            num_epochs: Optional[int],
            teach: Optional[bool] = False,
            performance_save_threshold: float = 1.0,
            teaching_csv_folder_name: str = "teach_arch",
    ):
        best_organisms = []
        ecosystem = Ecosystem(
            neuroevolution=self.neuroevolution,
        )
        good_archs = []
        good_archs_names = []
        for i in tqdm(range(1, self.neuroevolution.generations + 1), "generation"):
            population = ecosystem.generation(
                population=population,
                labels=self.neuroevolution.labels,
                generation=i,
            )
            this_generation_best = population[0]
            calc_structural_features = CalcStructuralFeatures(
                organism=this_generation_best,
            )
            calc_structural_features.add_modularity()
            best_organisms.append(this_generation_best)
            if i % self.save_every == 0:
                save_to_pkl(
                    best_folder=self.best_folder,
                    this_generation_best=this_generation_best,
                    generation=i,
                )
                print_log(
                    neuroevolution=self.neuroevolution,
                    this_generation_best=this_generation_best,
                    generation=i,
                )
                if performance_save_threshold - 0.5 <= this_generation_best.performance < performance_save_threshold + 0.5:
                    save_to_pkl(
                        best_folder=f"{self.base_path}/successful_results_5",
                        this_generation_best=this_generation_best,
                        generation=f"{self.experiment_name}_{i}",
                    )
                    if (
                            not good_archs
                            or (good_archs[-1].modularity != this_generation_best.modularity
                                and good_archs[-1].connection_cost != this_generation_best.connection_cost)
                    ):
                        good_archs.append(this_generation_best)
                        good_archs_names.append(f'best_{self.experiment_name}_{i}')
        export_results_to_csv(
            organisms=best_organisms,
            experiment_folder=self.current_experiment_folder,
        )
        if teach:
            self._teach_good_arch(
                good_archs=good_archs,
                good_archs_names=good_archs_names,
                num_exp_per_arch=num_exp_per_arch,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                activate=activate,
                model_cls=model_cls,
                teach_arch_folder_name=teaching_csv_folder_name,
            )

    def learn_evolutionarily_for_teachability(
            self,
            population: List[Organism],
    ):
        best_organisms = []
        ecosystem = Ecosystem(
            neuroevolution=self.neuroevolution,
        )
        good_archs = []
        for i in tqdm(range(1, self.neuroevolution.generations + 1), "generation"):
            population = ecosystem.generation(
                population=population,
                labels=self.neuroevolution.labels,
                generation=i,
            )
            this_generation_best = population[0]
            calc_structural_features = CalcStructuralFeatures(
                organism=this_generation_best,
            )
            calc_structural_features.add_modularity()
            best_organisms.append(this_generation_best)
            if i % self.save_every == 0:
                save_to_pkl(
                    best_folder=self.best_folder,
                    this_generation_best=this_generation_best,
                    generation=i,
                )
                print_log(
                    neuroevolution=self.neuroevolution,
                    this_generation_best=this_generation_best,
                    generation=i,
                )
                if this_generation_best.performance == 1:
                    if (
                            not good_archs
                            or good_archs[-1].modularity != this_generation_best.modularity
                    ):
                        save_to_pkl(
                            best_folder=f"{self.base_path}/successful_results_4",
                            this_generation_best=this_generation_best,
                            generation=f"{self.experiment_name}_{i}",
                        )
                        good_archs.append(this_generation_best)
        export_results_to_csv(
            organisms=best_organisms,
            experiment_folder=self.current_experiment_folder,
        )

    def learn_evolutionarily_for_teachability_treshold_with_teaching(
            self,
            population: List[Organism],
            num_exp_per_arch: Optional[int],
            learning_rate: Optional[float],
            num_epochs: Optional[int],
            model_cls: Optional[nn.Module],
            activate: Optional[str],
            teach: Optional[bool] = False,
            performance_save_threshold: float = 1.0,
            model_folder_name: str = 'successful_results',
            teaching_csv_folder_name: str = "teach_arch",
    ):
        ecosystem = Ecosystem(
            neuroevolution=self.neuroevolution,
        )
        good_archs = []
        good_archs_names = []
        for i in tqdm(range(1, self.neuroevolution.generations + 1), "generation"):
            population = ecosystem.generation(
                population=population,
                labels=self.neuroevolution.labels,
                generation=i,
            )
            if i % self.save_every == 0:
                for organism in population:
                    if performance_save_threshold - 0.5 <= organism.performance < performance_save_threshold + 0.5:
                        calc_structural_features = CalcStructuralFeatures(
                            organism=organism,
                        )
                        calc_structural_features.add_modularity()
                        if (
                                not good_archs
                                or (good_archs[-1].modularity != organism.modularity
                                    and good_archs[-1].connection_cost != organism.connection_cost)
                        ):
                            save_to_pkl(
                                best_folder=f"{self.base_path}/{model_folder_name}",
                                this_generation_best=organism,
                                generation=f"{self.experiment_name}_{i}"
                            )
                            good_archs.append(organism)
                            good_archs_names.append(f'best_{self.experiment_name}_{i}')
                            break
        if teach:
            self._teach_good_arch(
                good_archs=good_archs,
                good_archs_names=good_archs_names,
                num_exp_per_arch=num_exp_per_arch,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                teach_arch_folder_name=teaching_csv_folder_name,
                activate=activate,
                model_cls=model_cls,
            )
