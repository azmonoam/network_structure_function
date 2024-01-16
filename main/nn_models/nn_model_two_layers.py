from typing import List

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from nn_models.nn_model import NeuralNetModel


class FeedforwardNeuralNetModelTwoLayers(NeuralNetModel):
    def __init__(
            self,
            weight_mask: List[torch.Tensor],
            bias_mask: List[torch.Tensor],
            dimensions: List[int],
            reinitiate: bool,
            activate: str,
    ):
        super(FeedforwardNeuralNetModelTwoLayers, self).__init__(weight_mask, bias_mask, dimensions, reinitiate,
                                                                 activate)
        self.liner_input_to_hidden = prune.custom_from_mask(
            nn.Linear(dimensions[0], dimensions[1]),
            name='weight',
            mask=weight_mask[0].T,
        )
        self.liner_input_to_hidden = prune.custom_from_mask(
            self.liner_input_to_hidden,
            name='bias',
            mask=bias_mask[0],
        )
        self.liner_hidden_to_output = prune.custom_from_mask(
            nn.Linear(dimensions[1], dimensions[2]),
            name='weight',
            mask=weight_mask[1].T,
        )
        self.liner_hidden_to_output = prune.custom_from_mask(
            self.liner_hidden_to_output,
            name='bias',
            mask=bias_mask[1],
        )


    def forward(
            self,
            inputs: torch.Tensor,
    ) -> torch.Tensor:
        out = self.liner_input_to_hidden(inputs)
        out = self.activate(out)
        out = self.liner_hidden_to_output(out)
        return out

    def _update_initial_params(self):
        self.liner_input_to_hidden.weight.data = torch.nn.init.xavier_uniform_(
            self.liner_input_to_hidden.weight.data)
        self.liner_input_to_hidden.bias.data.uniform_(-2.0, 2.0)
        self.liner_hidden_to_output.weight.data = torch.nn.init.xavier_uniform_(
            self.liner_hidden_to_output.weight.data)
        self.liner_hidden_to_output.bias.data.uniform_(-2.0, 2.0)
