from typing import List

import torch
import torch.nn as nn


class NeuralNetModel(nn.Module):
    def __init__(
            self,
            weight_mask: List[torch.Tensor],
            bias_mask: List[torch.Tensor],
            dimensions: List[int],
            reinitiate: bool,
            activate: str,
    ):
        super(NeuralNetModel, self).__init__()
        if activate == 'tanh':
            self.activate = nn.Tanh()
        elif activate =='logistic':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(
            self,
            inputs: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @classmethod
    def _update_initial_params(self):
        pass
