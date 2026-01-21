import logging
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

import gpytorch
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel, SpectralMixtureKernel

logger = logging.getLogger()


class SMComponentKernel(Kernel):

    is_stationary = True
    def __init__(
        self,
        ard_num_dims: Optional[int] = 1,
        batch_shape: Optional[torch.Size] = torch.Size([]),
        mixture_scale_prior: Optional[Prior] = None,
        mixture_scale_constraint: Optional[Interval] = None,
        mixture_mean_prior: Optional[Prior] = None,
        mixture_mean_constraint: Optional[Interval] = None,
        mixture_weight_prior: Optional[Prior] = None,
        mixture_weight_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        
        super(SMComponentKernel, self).__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)

    
        self.register_parameter(
            name="raw_mixture_weight", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_parameter(name="raw_mixture_mean", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        self.register_parameter(name="raw_mixture_scale", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        if mixture_scale_constraint is None:
            mixture_scale_constraint = Positive()

        if mixture_mean_constraint is None:
            mixture_mean_constraint = Positive()

        if mixture_weight_constraint is None:
            mixture_weight_constraint = Positive()

        if mixture_scale_prior is not None:
            if not isinstance(mixture_scale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(mixture_scale_prior).__name__)
            self.register_prior(
                "mixture_scale_prior",
                mixture_scale_prior,
                lambda m: m.mixture_scale,
                lambda m, v: m._set_mixture_scale(v),
            )

        if mixture_mean_prior is not None:
            if not isinstance(mixture_mean_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(mixture_mean_prior).__name__)
            self.register_prior(
                "mixture_mean_prior",
                mixture_mean_prior,
                lambda m: m.mixture_mean,
                lambda m, v: m._set_mixture_mean(v),
            )

        if mixture_weight_prior is not None:
            if not isinstance(mixture_weight_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(mixture_weight_prior).__name__)
            self.register_prior(
                "mixture_weight_prior",
                mixture_weight_prior,
                lambda m: m.mixture_weight,
                lambda m, v: m._set_mixture_weight(v),
            )
        
        self.register_constraint("raw_mixture_scale", mixture_scale_constraint)
        self.register_constraint("raw_mixture_mean", mixture_mean_constraint)
        self.register_constraint("raw_mixture_weight", mixture_weight_constraint)

    @property
    def mixture_scale(self):
        return self.raw_mixture_scale_constraint.transform(self.raw_mixture_scale)

    @mixture_scale.setter
    def mixture_scale(self, value: Union[torch.Tensor, float]):
        self._set_mixture_scale(value)

    def _set_mixture_scale(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_scale)
        self.initialize(raw_mixture_scale=self.raw_mixture_scale_constraint.inverse_transform(value))

    @property
    def mixture_mean(self):
        return self.raw_mixture_mean_constraint.transform(self.raw_mixture_mean)

    @mixture_mean.setter
    def mixture_mean(self, value: Union[torch.Tensor, float]):
        self._set_mixture_mean(value)

    def _set_mixture_mean(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_mean)
        self.initialize(raw_mixture_mean=self.raw_mixture_mean_constraint.inverse_transform(value))

    @property
    def mixture_weight(self):
        return self.raw_mixture_weight_constraint.transform(self.raw_mixture_weight)

    @mixture_weight.setter
    def mixture_weight(self, value: Union[torch.Tensor, float]):
        self._set_mixture_weight(value)

    def _set_mixture_weight(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weight)
        self.initialize(raw_mixture_weight=self.raw_mixture_weight_constraint.inverse_transform(value))


    def import_params(self, sm: SpectralMixtureKernel, idx: int):
        self.mixture_scale
        self.mixture_mean
        self.mixture_weight


        
    def forward(self, x1: Tensor, x2: Tensor, **params):
        x1_exp = x1 * self.mixture_scale
        x2_exp = x2 * self.mixture_scale
        x1_cos = x1 * self.mixture_mean
        x2_cos = x2 * self.mixture_mean

        exp_term = (x1_exp - x2_exp).pow_(2).mul_(-2 * math.pi**2)
        cos_term = (x1_cos - x2_cos).mul_(2 * math.pi)
        res = exp_term.exp_() * cos_term.cos_()
        return res



