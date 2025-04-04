# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.init as init

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import Output, StudentTOutput
from gluonts.torch.util import weighted_average


def mean_abs_scaling(seq, min_scale=1e-5):
    return seq.abs().mean(1).clamp(min_scale, None).unsqueeze(1)

def mean_demand_scaling(seq, min_scale=1):
    return (torch.nanmean(seq.masked_fill(seq == 0, float('nan')), dim=1)
            .masked_fill(torch.isnan(torch.nanmean(seq.masked_fill(seq == 0, float('nan')), dim=1)), min_scale)
            .unsqueeze(1))


def make_linear_layer(dim_in, dim_out):
    lin = nn.Linear(dim_in, dim_out)
    torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
    torch.nn.init.zeros_(lin.bias)
    return lin


class SimpleFeedForwardModel(nn.Module):
    """
    Module implementing a feed-forward model for forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dimensions
        Size of hidden layers in the feed-forward network.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    batch_norm
        Whether to apply batch normalization. Default: ``False``.
    """

    @validated()
    def __init__(
        self,
        scale: Optional[str],
        prediction_length: int,
        context_length: int,
        hidden_dimensions: Optional[List[int]] = None,
        distr_output: Output = StudentTOutput(),
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert hidden_dimensions is None or len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = (
            hidden_dimensions if hidden_dimensions is not None else [20, 20]
        )
        self.distr_output = distr_output
        self.batch_norm = batch_norm

        dimensions = [context_length] + self.hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [make_linear_layer(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(
            make_linear_layer(
                dimensions[-1], prediction_length * self.hidden_dimensions[-1]
            )
        )
        self.scale = scale
        self.nn = nn.Sequential(*modules)
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
        self.args_proj = self.distr_output.get_args_proj(
            self.hidden_dimensions[-1]
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        if self.scale == "mean-demand":
            s = mean_demand_scaling(past_target)  
        elif self.scale == "mean":
            s = mean_abs_scaling(past_target)     
        else:
            s = torch.ones((past_target.shape[0],1))
        scaled_context = past_target / s
        nn_out = self.nn(scaled_context)
        nn_out_reshaped = nn_out.reshape(
            -1, self.prediction_length, self.hidden_dimensions[-1]
        )
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, torch.zeros((past_target.shape[0],1), device=s.device), s

    def loss(
        self,
        past_target: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        distr_args, loc, scale = self(past_target=past_target)
        if (torch.isnan(distr_args[0]).sum()):
            print('hey')
        loss = self.distr_output.loss(
            target=future_target,
            distr_args=distr_args,
            loc=loc,
            scale=scale,
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
