from typing import Optional, List, Tuple


import torch
import torch.nn as nn
import numpy as np
from nn_models.nn_model_four_layers import FeedforwardNeuralNetModelFourLayers
from organism import Organism
from tasks import Task
from utils.main_utils import get_unconnected_nodes


class TeachArch:
    def __init__(
            self,
            task: Task,
            model_cls: nn.Module = FeedforwardNeuralNetModelFourLayers,
            learning_rate: float = 0.1,
            num_epochs: int = 100,
            reinitiate: bool = True,
            activate: str = 'tanh',
            optimizer: torch.optim.Optimizer = torch.optim.SGD,
            flatten: bool = False,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.calc_accuracy = task.calc_accuracy
        self.train_loader, self.test_loader = task.data_fetcher.get_data_loaders()
        self.criterion = nn.CrossEntropyLoss()
        self.reinitiate = reinitiate
        self.activate = activate
        self.model_cls = model_cls
        self.optimizer = optimizer
        self.flatten = flatten
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'******* Running on {self.device} *******', flush=True)

    @staticmethod
    def _get_weights_mask(
            organism: Organism,
            weight: Optional[str],
    ):
        layers = organism.convert_network_to_list_of_layers_arrays(
            weight=weight,
        )
        return [
            torch.tensor(layer)
            for layer in layers
        ]

    @staticmethod
    def _get_bias_mask(
            organism: Organism,
    ):
        torch_biases = [torch.ones((i,)) for i in organism.dimensions[1:]]
        unconnected_node = get_unconnected_nodes(
            network=organism.network,
            input_layer=0,
        )
        for neuron in unconnected_node:
            for layer_ind, layer in enumerate(organism.layer_neuron_idx_mapping[1:]):
                for neuron_ind, neuron_in_layer in enumerate(layer):
                    if neuron_in_layer == neuron:
                        torch_biases[layer_ind][neuron_ind] = 0
        return torch_biases

    def _get_models_features(
            self,
            organism: Organism,
            weight: Optional[str] = 'weight',
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        weights_mask = self._get_weights_mask(
            organism=organism,
            weight=weight,
        )
        biases_mask = self._get_bias_mask(
            organism=organism,
        )
        return weights_mask, biases_mask

    def _build_model(
            self,
            organism: Organism,
            weights_mask: List[torch.Tensor],
            biases_mask: List[torch.Tensor],
    ):
        return self.model_cls(
            dimensions=organism.dimensions,
            weight_mask=weights_mask,
            bias_mask=biases_mask,
            reinitiate=self.reinitiate,
            activate=self.activate,
        )

    def _train_modal(
            self,
            model: nn.Module,
            optimizer: torch.optim,
    ):
        all_batches_losses = []
        for train_input, train_label in self.train_loader:
            train_input = train_input.to(self.device)
            train_label = train_label.to(self.device)
            optimizer.zero_grad()
            if self.flatten:
                train_input = train_input.reshape(-1, train_input.shape[-1] * train_input.shape[-2])
            outputs = model(train_input)
            loss = self.criterion(outputs, train_label)
            loss.backward()
            optimizer.step()
            all_batches_losses.append(loss.item())
        return optimizer, np.mean(all_batches_losses)

    def _test_model(
            self,
            model: nn.Module,
    ) -> float:
        correct = 0
        total = 0
        for test_input_list, test_label in self.test_loader:
            test_input_list = test_input_list.to(self.device)
            test_label = test_label.to(self.device)
            if self.flatten:
                test_input_list = test_input_list.reshape(-1, test_input_list.shape[-1] * test_input_list.shape[-2])
            test_outputs = model(test_input_list)
            total += test_label.size(0)
            _, predicted = torch.max(test_outputs.data, 1)
            correct += self.calc_accuracy(
                prediction=predicted,
                label=test_label,
            )
        return correct / total

    def back_prop_teach_arch(
            self,
            organism: Organism,
    ) -> float:
        weights_mask, biases_mask = self._get_models_features(
            organism=organism,
            weight=None,
        )
        model = self._build_model(
            organism=organism,
            weights_mask=weights_mask,
            biases_mask=biases_mask,
        ).to(self.device)
        model.train()
        optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
        all_epochs_loss = []
        for epoch in range(self.num_epochs):
            optimizer, loss = self._train_modal(
                model=model,
                optimizer=optimizer,
            )
            all_epochs_loss.append(loss)
        accuracy = self._test_model(
            model=model,
        )
        return accuracy
