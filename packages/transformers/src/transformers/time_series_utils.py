# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Time series distributional output classes and utilities.
"""
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributions import (
    AffineTransform,
    Distribution,
    Independent,
    NegativeBinomial,
    Normal,
    StudentT,
    Poisson,
    TransformedDistribution,
)

from .distributions_utils import (
    Tweedie, 
    FixedDispersionTweedie, 
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)

class AffineTransformed(TransformedDistribution):
    def __init__(self, base_distribution: Distribution, loc=None, scale=None, event_dim=0):
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc

        super().__init__(base_distribution, [AffineTransform(loc=self.loc, scale=self.scale, event_dim=event_dim)])

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale**2

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()


class ParameterProjection(nn.Module):
    def __init__(
        self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[torch.Tensor]], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x, *args):
        return self.function(x, *args)


class DistributionOutput:
    distribution_class: type
    in_features: int
    args_dim: Dict[str, int]

    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distribution_class(*distr_args)
        else:
            return Independent(self.distribution_class(*distr_args), 1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return AffineTransformed(distr, loc=loc, scale=scale, event_dim=self.event_dim)

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions that this object constructs.
        """
        return () if self.dim == 1 else (self.dim,)

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object
        constructs.
        """
        return len(self.event_shape)

    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
        return 0.0

    def get_parameter_projection(self, in_features: int) -> nn.Module:
        r"""
        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.
        """
        return ParameterProjection(
            in_features=in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
        )

    def domain_map(self, *args: torch.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the
        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a
        distribution of the right event_shape.
        """
        raise NotImplementedError()

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0


class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """

    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distribution_class: type = StudentT

    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        df = 2.0 + cls.squareplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """

    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distribution_class: type = Normal

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        return loc.squeeze(-1), scale.squeeze(-1)


class NegativeBinomialOutput(DistributionOutput):
    """
    Negative Binomial distribution output class.
    """

    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    distribution_class: type = NegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        total_count = cls.squareplus(total_count)
        return total_count.squeeze(-1), logits.squeeze(-1)

    def _base_distribution(self, distr_args) -> Distribution:
        total_count, logits = distr_args
        if self.dim == 1:
            return self.distribution_class(total_count=total_count, logits=logits)
        else:
            return Independent(self.distribution_class(total_count=total_count, logits=logits), 1)

    # Overwrites the parent class method. We cannot scale using the affine
    # transformation since negative binomial should return integers. Instead
    # we scale the parameters.
    def distribution(
        self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            # See scaling property of Gamma.
            logits += scale.log()

        return self._base_distribution((total_count, logits))
    

class TweedieOutput(DistributionOutput):

    args_dim: Dict[str, int] = {"mu": 1, "phi": 1, "rho":1}
    distribution_class: type = Tweedie

    @classmethod
    def domain_map(cls, mu: torch.Tensor, phi: torch.Tensor, rho: torch.Tensor):
        mu = cls.squareplus(mu).clamp_min(torch.finfo(mu.dtype).eps)
        phi = cls.squareplus(phi).clamp_min(torch.finfo(phi.dtype).eps)
        rho = (1+rho.sigmoid()).clamp(1+torch.finfo(rho.dtype).eps, 2-torch.finfo(rho.dtype).eps)
        
        return mu.squeeze(-1), phi.squeeze(-1), rho.squeeze(-1)
    
    def _base_distribution(self, distr_args) -> Distribution:
        mu, phi, rho = distr_args
        if self.dim == 1:
            return self.distribution_class(mu=mu, phi=phi, rho=rho)
        else:
            return Independent(self.distribution_class(mu=mu, phi=phi, rho=rho), 1)
        
    @property
    def event_shape(self) -> Tuple:
        return ()
    

class FixedDispersionTweedieOutput(DistributionOutput):

    args_dim: Dict[str, int] = {"mu": 1, "rho":1}
    distribution_class: type = FixedDispersionTweedie

    @classmethod
    def domain_map(cls, mu: torch.Tensor, rho: torch.Tensor):
        mu = cls.squareplus(mu).clamp_min(torch.finfo(mu.dtype).eps)
        rho = (1+rho.sigmoid()).clamp(1+torch.finfo(rho.dtype).eps, 2-torch.finfo(rho.dtype).eps)
        
        return mu.squeeze(-1), rho.squeeze(-1)
    
    def _base_distribution(self, distr_args) -> Distribution:
        mu, rho = distr_args
        if self.dim == 1:
            return self.distribution_class(mu=mu, rho=rho)
        else:
            return Independent(self.distribution_class(mu=mu, rho=rho), 1)
        
    @property
    def event_shape(self) -> Tuple:
        return ()


class PoissonOutput(DistributionOutput):

    args_dim: Dict[str, int] = {"rate": 1}
    distribution_class: type = Poisson

    @classmethod
    def domain_map(cls, rate: torch.Tensor):
        rate = cls.squareplus(rate).clamp_min(torch.finfo(rate.dtype).eps)
        return rate.squeeze(-1)
    
    def _base_distribution(self, distr_args) -> Distribution:
        rate = distr_args
        if self.dim == 1:
            return self.distribution_class(rate=rate)
        else:
            return Independent(self.distribution_class(rate=rate), 1)

    def distribution(
        self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        rate = distr_args

        if scale is not None:   
            rate = rate * scale

        return self._base_distribution((rate))
    

class ZeroInflatedPoissonOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1, "p":1}
    distribution_class: type = ZeroInflatedPoisson

    @classmethod
    def domain_map(cls, rate: torch.Tensor, p: torch.Tensor):
        rate = cls.squareplus(rate).clamp_min(torch.finfo(rate.dtype).eps)
        p = p.sigmoid().clamp(torch.finfo(p.dtype).eps, 1-torch.finfo(p.dtype).eps)
        return rate.squeeze(-1), p.squeeze(-1)
    
    def _base_distribution(self, distr_args) -> Distribution:
        rate, p = distr_args
        if self.dim == 1:
            return self.distribution_class(rate=rate, p=p)
        else:
            return Independent(self.distribution_class(rate=rate, p=p), 1)
        
    def distribution(
        self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        rate, p = distr_args

        if scale is not None:   
            rate = rate * scale

        return self._base_distribution((rate, p))
  

class ZeroInflatedNegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"total_count": 1, "probs":1, "p_zero":1}
    distribution_class: type = ZeroInflatedNegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, probs: torch.Tensor, p_zero: torch.Tensor):
        total_count = cls.squareplus(total_count).clamp_min(torch.finfo(total_count.dtype).eps)
        probs = probs.sigmoid().clamp(torch.finfo(probs.dtype).eps, 1-torch.finfo(probs.dtype).eps)
        p_zero = probs.sigmoid().clamp(torch.finfo(p_zero.dtype).eps, 1-torch.finfo(p_zero.dtype).eps)
        return  total_count.squeeze(-1), probs.squeeze(-1), p_zero.squeeze(-1)
    
    def _base_distribution(self, distr_args) -> Distribution:
        total_count, probs, p_zero = distr_args
        if self.dim == 1:
            return self.distribution_class(total_count=total_count, probs=probs, p_zero=p_zero)
        else:
            return Independent(self.distribution_class(total_count=total_count, probs=probs, p_zero=p_zero), 1)
        
    def distribution(
        self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        total_count, probs, p_zero = distr_args

        if scale is not None:   
            mu = total_count*probs/(1-probs)
            logits = torch.logit(self.probs)
            logits += ((scale*(1. + mu) -1.)/mu).log()
            probs = logits.exp()/(1. + logits.exp())

        return self._base_distribution((total_count, probs, p_zero))