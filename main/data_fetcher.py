import itertools
import random
from typing import List, Tuple, Union

import torch
import torchvision
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader

Logical = Union[bool, int]
RANDOM_SEED = 1994

class DataFetcher:
    def __init__(
            self,
            task,
    ):
        self.task = task

    def get_inputs(
            self,
    ) -> List[List[int]]:
        return [
            list(i)
            for i in itertools.product([0, 1], repeat=self.task.input_dim)
        ]

    def get_labels(
            self,
            inputs: List[List[Logical]],
    ) -> List[Logical]:
        if isinstance(self.task.rule, list):
            self.task.rule = self.task.rule[0]
        return [
            self.task.solve_task(
                input_pattern=input_pattern,
            ) * 1
            for input_pattern in inputs
        ]

    def crete_data_set_for_loader(
            self,
            inputs: List[List[Logical]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.tensor(
                    input_pattern,
                    dtype=torch.float
                ),
                self._create_label(
                    input_pattern=input_pattern,
                )
            )
            for input_pattern in inputs
        ]

    def _create_label(
            self,
            input_pattern: List[Logical],
    ):
        if self.task.solve_task(
                input_pattern=input_pattern,
        ) * 1 == 1:
            return torch.Tensor([0, 1])
        return torch.Tensor([1, 0])

    def get_data_loaders(
            self,
    ) -> Tuple[DataLoader, DataLoader]:
        inputs = self.get_inputs()
        dataset = self.crete_data_set_for_loader(
            inputs=inputs,
        )
        batch_size = int(len(dataset))
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader, train_loader


class DigitDataFetcher(DataFetcher):
    def __init__(
            self,
            task,
            train_proportion: float = 0.8
    ):
        super().__init__(task)
        self.task = task
        self.data = load_digits()
        self.train_proportion = train_proportion

    def get_inputs(
            self,
    ) -> List[List[int]]:
        return self.data.data

    def get_labels(
            self,
            inputs: List[List[Logical]],
    ) -> List[Logical]:
        raise NotImplementedError

    def _create_label(
            self,
            input_pattern: List[Logical],
    ):
        raise NotImplementedError

    def _get_labels(
            self,
            ind: int
    ) -> torch.Tensor:
        label = torch.zeros(10)
        label[self.data.target[ind]] = 1
        return label

    def crete_data_set_for_loader(
            self,
            inputs: List[List[Logical]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.tensor(
                    input_pattern,
                    dtype=torch.float
                ),
                self._get_labels(
                    ind=ind,
                )
            )
            for ind, input_pattern in enumerate(inputs)
        ]

    def get_data_loaders(
            self,
    ) -> Tuple[DataLoader, DataLoader]:
        inputs = self.get_inputs()
        dataset = self.crete_data_set_for_loader(
            inputs=inputs,
        )
        train_num_samples = round(len(dataset) * self.train_proportion)
        random.seed(RANDOM_SEED)
        random.shuffle(dataset)
        train_loader = DataLoader(
            dataset=dataset[:train_num_samples],
            batch_size=self.task.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=dataset[train_num_samples:],
            batch_size=self.task.batch_size,
            shuffle=False,
        )
        return train_loader, test_loader


class PytorchDataFetcher:
    def __init__(
            self,
            task,
    ):
        self.task = task

    @staticmethod
    def crete_data_set_for_loader() -> Tuple[torchvision.datasets.mnist.MNIST, torchvision.datasets.mnist.MNIST]:
        train_data = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_data = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            transform=torchvision.transforms.ToTensor(),
        )
        return train_data, test_data

    def get_data_loaders(
            self,
    ) -> Tuple[DataLoader, DataLoader]:
        train_data, test_data = self.crete_data_set_for_loader()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.task.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=self.task.batch_size,
            shuffle=False,
        )
        return train_loader, test_loader
