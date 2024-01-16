import torch.nn as nn
import torch.nn.functional as functional
import torch
from typing import List


class SmallNN(nn.Module):
    def __init__(
            self,
            use_cuda: bool,
            layers_sized: List[int]
    ):
        # options for dtype are: float(32-bit), double(64-bit), half(16-bit). default is float
        #dtype = torch.half if use_cuda else torch.float
        dtype = torch.float
        #   in_features: number of features of input.
        super(SmallNN, self).__init__()
        self.fc1 = nn.Linear(layers_sized[0], layers_sized[1]).type(dtype)
        self.fc2 = nn.Linear(layers_sized[1], layers_sized[2]).type(dtype)
        self.activation = functional.relu

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)
