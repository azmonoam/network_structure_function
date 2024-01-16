import multiprocessing
import subprocess
from typing import Tuple, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed

from new_organism import Organism
from nn_models.nn_model_four_layers import FeedforwardNeuralNetModelFourLayers
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from tasks import Task
from teach_arch import TeachArch


class TeachArchMultiTime(TeachArch):
    def __init__(
            self,
            output_path: str,
            exp_name: str,
            task: Task,
            model_cls: nn.Module = FeedforwardNeuralNetModelFourLayers,
            learning_rate: float = 0.1,
            num_epochs: int = 100,
            reinitiate: bool = True,
            num_exp_per_arch: int = 100,
            activate: str = 'tanh',
            optimizer: torch.optim.Optimizer = torch.optim.SGD,
            flatten: bool = False,
            test_every: Optional[int] = None,
            n_threads: Optional[int] = None
    ):
        super().__init__(task, model_cls, learning_rate, num_epochs, reinitiate, activate, optimizer, flatten)
        self.num_exp_per_arch = num_exp_per_arch
        self.output_path = output_path
        self.exp_name = exp_name

        self.num_cores = n_threads
        if not self.num_cores:
            self.num_cores = multiprocessing.cpu_count()
        print(f"-- using {self.num_cores} cores --")

        if not test_every:
            test_every = self.num_epochs * 0.05
        self.test_every = test_every

    def teach_arch_many_times(
            self,
            organism: Organism,
            return_all_the_way: bool = False,
            min_loss_limit: Optional[float] = None,
    ):
        organism = self._setup_stractural_features(
            organism=organism,
        )
        weights_mask, biases_mask = self._get_models_features(
            organism=organism,
            weight=None,
        )
        performances = []
        losses = []
        experiments = []
        iterations = []
        for exp in range(self.num_exp_per_arch):
            if return_all_the_way:
                accuracies, exp_losses, exps, epochs = self._learn(
                    exp=exp,
                    organism=organism,
                    weights_mask=weights_mask,
                    biases_mask=biases_mask,
                    return_all_the_way=return_all_the_way,
                    min_loss_limit=min_loss_limit,
                )
                performances += accuracies
                losses += exp_losses
                experiments += exps
                iterations += epochs
            else:
                accuracy, loss, exp, epoch = self._learn(
                    exp=exp,
                    organism=organism,
                    weights_mask=weights_mask,
                    biases_mask=biases_mask,
                    return_all_the_way=return_all_the_way,
                    min_loss_limit=min_loss_limit,
                )
                performances.append(accuracy)
                losses.append(loss)
                experiments.append(exp)
                iterations.append(epoch)
        length = len(experiments)
        pd.DataFrame.from_dict(
            {
                'exp_name': [self.exp_name] * length,
                'performance': performances,
                'loss': losses,
                'experiment': experiments,
                'iterations': iterations,
                'modularity': [organism.structural_features.modularity.modularity] * length,
                'num_connections': [organism.structural_features.connectivity.num_connections] * length,
                'entropy': [organism.structural_features.entropy.entropy] * length,
                'normed_entropy': [organism.structural_features.entropy.normed_entropy] * length,
                'connectivity_ratio': [round(organism.structural_features.connectivity.connectivity_ratio, 2)] * length,
            }
        ).to_csv(self.output_path)

    def teach_arch_many_times_parallel(
            self,
            organism: Organism,
            return_all_the_way: bool = False,
            min_loss_limit: Optional[float] = None,
    ):
        organism = self._setup_stractural_features(
            organism=organism,
        )
        performances = []
        losses = []
        experiments = []
        iterations = []
        res = self._learn_parallel_wrapper(
            organism=organism,
            return_all_the_way=return_all_the_way,
            min_loss_limit=min_loss_limit,
        )
        if return_all_the_way:
            for accuracies, exp_losses, exps, epochs in res:
                performances += accuracies
                losses += exp_losses
                experiments += exps
                iterations += epochs
            length = len(experiments)
            pd.DataFrame.from_dict(
                {
                    'exp_name': [self.exp_name] * length,
                    'performance': performances,
                    'loss': losses,
                    'experiment': experiments,
                    'iterations': iterations,
                }
            ).to_csv(self.output_path)
        else:
            for accuracy, loss, exp, epoch in res:
                performances.append(accuracy)
                losses.append(loss)
                experiments.append(exp)
                iterations.append(epoch)
            length = len(experiments)
            pd.DataFrame.from_dict(
                {
                    'exp_name': [self.exp_name] * length,
                    'performance': performances,
                    'loss': losses,
                    'experiment': experiments,
                    'iterations': iterations,
                    'modularity': [organism.structural_features.modularity.modularity] * length,
                    'num_connections': [organism.structural_features.connectivity.num_connections] * length,
                    'entropy': [organism.structural_features.entropy.entropy] * length,
                    'normed_entropy': [organism.structural_features.entropy.normed_entropy] * length,
                    'connectivity_ratio': [round(organism.structural_features.connectivity.connectivity_ratio, 2)] * length,
                }
            ).to_csv(self.output_path)

    def _learn(
            self,
            exp: int,
            organism: Organism,
            weights_mask: List[torch.Tensor],
            biases_mask: List[torch.Tensor],
            return_all_the_way: bool,
            min_loss_limit: Optional[float],
    ):
        model = self._build_model(
            organism=organism,
            weights_mask=weights_mask,
            biases_mask=biases_mask,
        ).to(self.device)
        #print(f'model: {next(model.parameters()).device}')
        model.train()
        optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
        losses = []
        accuracies = []
        for epoch in range(self.num_epochs + 1):
            optimizer, loss = self._train_modal(
                model=model,
                optimizer=optimizer,
            )
            if epoch % self.test_every == 0:
                accuracy = self._test_model(
                    model=model,
                )
                losses.append(loss.item())
                accuracies.append(accuracy)
                print(
                    f'Experiment: {exp}. Iteration: {epoch}. '
                    f'Loss: {loss.item():.3f}. Accuracy: {accuracy:.3f}.')

                if epoch == self.num_epochs:
                    break
                if not return_all_the_way and self._converged_early(
                    losses=losses,
                    accuracies=accuracies,
                ):
                    break
                if min_loss_limit is not None and loss < min_loss_limit:
                    break
        if return_all_the_way:
            epochs_array = np.arange(0, self.num_epochs + self.test_every, self.test_every,  dtype=int)
            epochs = epochs_array.tolist()
            exps = [exp] * len(epochs)
            while len(accuracies) != len(epochs):
                accuracies.append(accuracies[-1])
                losses.append(losses[-1])
            return accuracies, losses, exps, epochs
        return accuracy, loss.item(), exp, epoch

    def _converged_early(
            self,
            losses: List[float],
            accuracies: List[float],
    ) -> bool:
        if accuracies[-1] == 1.0:
            return True
        no_change_limit = int((self.num_epochs / self.test_every) * 0.25)
        if len(losses) > no_change_limit:
            l = losses[-no_change_limit:]
            if sum(
                    int(l[i] >= l[i - 1]) for i in range(1, len(l))
            ) == (len(l) - 1):
                return True
        no_change_limit = int((self.num_epochs / self.test_every) * 0.35)
        if len(accuracies) > no_change_limit:
            a = accuracies[-no_change_limit:]
            if sum(
                    int(a[i] <= a[i - 1]) for i in range(1, len(a))
            ) == (len(a) - 1):
                return True
        return False

    def _learn_parallel_wrapper(
            self,
            organism: Organism,
            return_all_the_way: bool,
            min_loss_limit: Optional[float],
    ) -> Tuple[float, float, int, int]:
        weights_mask, biases_mask = self._get_models_features(
            organism=organism,
            weight=None,
        )
        print('--- running parallel ---')
        try:
            NUM_CORES = int(subprocess.run('nproc', capture_output=True).stdout)
        except FileNotFoundError:
            NUM_CORES = self.num_cores
        print(f'--- NUM_CORES: {NUM_CORES} ---')
        return Parallel(n_jobs=NUM_CORES)(
            delayed
            (self._learn)(exp, organism, weights_mask, biases_mask, return_all_the_way, min_loss_limit)
            for exp in range(self.num_exp_per_arch)
        )

    def _setup_stractural_features(
            self,
            organism: Organism,
    ) -> Organism:
        calc_structural_features = CalcStructuralFeatures(
            organism=organism,
        )
        if organism.structural_features.modularity is None:
            organism.structural_features.modularity = calc_structural_features.get_modularity()
        if organism.structural_features.connectivity is None:
            organism.structural_features.connectivity = calc_structural_features.get_connectivity()
        if organism.structural_features.entropy is None:
            organism.structural_features.entropy = calc_structural_features.get_entropy(
                num_connections=organism.structural_features.connectivity.num_connections,
            )
        return organism
