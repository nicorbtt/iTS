from numbers import Number
from typing import Dict, Tuple
from .distribution_output import DistributionOutput

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions import Poisson
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions import Poisson, Gamma



class ZeroInflatedPoisson(ExponentialFamily):

    arg_constraints = {"rate": constraints.nonnegative,
                       "p": constraints.interval(0,1)}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0
    
    @property
    def mean(self):
        return self.p*(self.rate-1)
        
    @property
    def variance(self):
        raise NotImplementedError()
    
    def __init__(self, rate, p, validate_args=None):
        self.rate, self.p = broadcast_all(rate, p)
        if isinstance(rate, Number) and isinstance(p, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.p.device)
        if self._validate_args:
            self._validate_sample(value)
            
        value, rate, p = broadcast_all(value, self.rate, self.p)
        
        log_p = torch.full(value.shape, torch.nan)

        zeros = value == 0
        non_zeros = ~zeros
        
        if torch.any(zeros):
            log_p[zeros] = torch.log(p[zeros])
        
        if torch.any(non_zeros):
            log_p[non_zeros] = torch.log(1-p[non_zeros]) + Poisson(rate[non_zeros]).log_prob(value[non_zeros] - 1)
        
        return log_p
    
    def rsample(self, sample_shape=torch.Size()):
        
        rate, p = broadcast_all(self.rate, self.p)

        with torch.no_grad():
            
            return torch.bernoulli(torch.broadcast_to(p, sample_shape + p.shape))*(1+Poisson(rate).sample(sample_shape))
        
    def cdf(self, value):
        
        raise NotImplementedError
    

class ZeroInflatedPoissonOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"rate": 1, "p":1}
    distr_cls: type = ZeroInflatedPoisson

    @classmethod
    def domain_map(cls, rate: torch.Tensor, p: torch.Tensor):
        rate = F.softplus(rate).clamp_min(torch.finfo(rate.dtype).eps)
        p = p.sigmoid().clamp(torch.finfo(p.dtype).eps, 1-torch.finfo(p.dtype).eps)
        return rate.squeeze(-1), p.squeeze(-1)
        
    @property
    def event_shape(self) -> Tuple:
        return ()