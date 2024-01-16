from typing import List, Optional

import torch.nn as nn
from tqdm import tqdm

from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from neuroevolution import Neuroevolution
from nn_models.nn_model_four_layers import FeedforwardNeuralNetModelFourLayers
from tasks import Task, RetinaTask
from teachable_ecosystem import TeachableEcosystem
from utils.set_up_population_utils import get_population_by_connectivity_ratio
from utils.main_utils import export_results_to_csv, save_to_pkl, print_log
from utils.main_utils import set_up_experiment_folders


def teach_fast(
        experiment_name: str,
        base_path: str,
        neuroevolution: Neuroevolution,
        learning_rate: float,
        num_epochs: int,
        reinitiate: bool,
        out_path: str,
        start_dimensions: Optional[List[int]],
        connectivity_ratio: Optional[float],
        save_every: int = 100,
        model_cls: nn.Module = FeedforwardNeuralNetModelFourLayers,
        task: Task = RetinaTask,
        activate: str = 'tanh',
):
    best_organisms = []
    best_folder, current_experiment_folder = set_up_experiment_folders(
        experiment_name=experiment_name,
        base_path=base_path,
    )

    population = get_population_by_connectivity_ratio(
        neuroevolution=neuroevolution,
        start_dimensions=start_dimensions,
        connectivity_ratio=connectivity_ratio,
    )
    ecosystem = TeachableEcosystem(
        neuroevolution=neuroevolution,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        reinitiate=reinitiate,
        model_cls=model_cls,
        task=task,
        activate=activate,
    )
    for i in tqdm(range(1, neuroevolution.generations + 1), "generation"):
        population = ecosystem.generation(
            population=population,
            labels=neuroevolution.labels,
            generation=i
        )
        this_generation_best = population[0]
        calc_structural_features = CalcStructuralFeatures(
            organism=this_generation_best,
        )
        calc_structural_features.add_modularity()
        best_organisms.append(this_generation_best)
        if i % save_every == 0:
            save_to_pkl(
                best_folder=best_folder,
                this_generation_best=this_generation_best,
                generation=i,
            )
            print_log(
                neuroevolution=neuroevolution,
                this_generation_best=this_generation_best,
                generation=i,
            )
    export_results_to_csv(
        organisms=best_organisms,
        experiment_folder=current_experiment_folder,
    )
    export_results_to_csv(
        organisms=best_organisms,
        experiment_folder=out_path,
        csv_name=experiment_name,
    )
