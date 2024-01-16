from typing import List, Callable, Optional

from data_fetcher import DataFetcher, PytorchDataFetcher, DigitDataFetcher
from logical_gates import LogicalGates


class Task:
    def __init__(
            self,
            input_dim: int,
            rule: Callable,
    ):
        self.communities_input_symmetry = None
        self.input_dim = input_dim
        self.rule = rule
        self.data_fetcher = DataFetcher(
            task=self,
        )
        self.logical_gates = LogicalGates()

    def solve_task(
            self,
            input_pattern: List[int],
    ) -> int:
        pass

    @staticmethod
    def calc_accuracy(
            prediction,
            label,
    ):
        return (prediction == label.argmax(axis=1)).sum().item()


class RetinaTask(Task):

    def __init__(
            self,
            input_dim: int,
            rule: Callable,
    ):
        super(RetinaTask, self).__init__(input_dim, rule)
        self.communities_input_symmetry = 2
        ALLOWED_PATTERNS = {
            8: {
                'right': [
                    [1, 1, 1, 1],
                    [1, 1, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 1],
                    [0, 0, 0, 1],
                    [1, 1, 0, 1],
                    [0, 1, 0, 0],
                    [0, 1, 1, 1]
                ],
                'left': [
                    [1, 1, 1, 1],
                    [1, 1, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 1],
                    [1, 0, 0, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 1, 1]
                ]
            },
            6: {
                'right': [
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 1, 1]
                ],
                'left': [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0]
                ]
            },
        }
        self.left_allowed_patterns = ALLOWED_PATTERNS[input_dim]['left']
        self.right_allowed_patterns = ALLOWED_PATTERNS[input_dim]['right']

    def solve_task(
            self,
            input_pattern: List[int],
    ) -> int:
        cutoff = self.input_dim // 2
        left = input_pattern[:cutoff] in self.left_allowed_patterns
        right = input_pattern[cutoff:] in self.right_allowed_patterns
        return self.rule(left, right)


class XorsTask(Task):

    def __init__(
            self,
            input_dim: int,
            rule: Callable,
    ):
        super(XorsTask, self).__init__(input_dim, rule)
        self.communities_input_symmetry = 3

    def solve_task(
            self,
            input_pattern: List[int],
    ) -> int:
        couples = [
            self.logical_gates.XOR(input_pattern[i], input_pattern[i + 1])
            for i in range(0, self.input_dim, 2)
        ]
        res = None
        for i in range(0, len(couples), 2):
            if res is None:
                res = (self.rule(couples[i], couples[i + 1]))
            else:
                res = self.rule(couples[i], res)
        return res


class DigitsTask(Task):

    def __init__(
            self,
            input_dim: int,
            rule: Optional[Callable] = None,
    ):
        super(DigitsTask, self).__init__(input_dim, rule)
        self.batch_size = 100000
        self.data_fetcher = DigitDataFetcher(
            task=self,
        )

    def solve_task(
            self,
            input_pattern: List[int],
    ) -> int:
        raise NotImplementedError


class MnistTask(Task):
    def __init__(
            self,
            input_dim: int,
            rule: Optional[Callable] = None,
    ):
        super(MnistTask, self).__init__(input_dim, rule)
        self.batch_size = 128
        self.data_fetcher = PytorchDataFetcher(
            task=self,
        )

    @staticmethod
    def calc_accuracy(
            prediction,
            label,
    ):
        return (prediction == label).sum().item()

    def solve_task(
            self,
            input_pattern: List[int],
    ) -> int:
        raise NotImplementedError
