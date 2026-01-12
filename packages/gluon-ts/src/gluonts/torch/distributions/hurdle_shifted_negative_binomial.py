from numbers import Number
from typing import Dict, Tuple, Optional
from .distribution_output import DistributionOutput

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions import NegativeBinomial, Distribution


class ZeroInflatedNegativeBinomial(ExponentialFamily):

    arg_constraints = {"total_count": constraints.nonnegative,
                       "logits": constraints.real,
                       "p_zero": constraints.interval(0,1)}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0
    
    @property
    def mean(self):
        raise NotImplementedError()
        
    @property
    def variance(self):
        raise NotImplementedError()
    
    def __init__(self, total_count, logits, p_zero, validate_args=None):
        self.total_count, self.logits, self.p_zero = broadcast_all(total_count, logits, p_zero)
        if isinstance(total_count, Number) and isinstance(logits, Number) and isinstance(p_zero, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.total_count.size()
        super().__init__(batch_shape, validate_args=validate_args)


    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.total_count.dtype, device=self.p_zero.device)
        if self._validate_args:
            self._validate_sample(value)
            
        value, total_count, logits, p_zero = broadcast_all(value, self.total_count, self.logits, self.p_zero)
        
        log_p = torch.full(value.shape, torch.nan, device = value.device)

        zeros = value == 0
        non_zeros = ~zeros
        
        if torch.any(zeros):
            log_p[zeros] = torch.log(p_zero[zeros])
        
        if torch.any(non_zeros):
            log_p[non_zeros] = torch.log(1-p_zero[non_zeros]) + NegativeBinomial(total_count=total_count[non_zeros], 
                                                                                 logits=logits[non_zeros]).log_prob(value[non_zeros] - 1)
        
        return log_p
    
    def rsample(self, sample_shape=torch.Size()):
        
        total_count, logits, p_zero = broadcast_all(self.total_count, self.logits, self.p_zero)

        with torch.no_grad():
            
            return torch.bernoulli(torch.broadcast_to(1-p_zero, sample_shape + p_zero.shape))*(1+NegativeBinomial(total_count=total_count, 
                                                                                                                  logits=logits).sample(sample_shape))
        
    

class ZeroInflatedNegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"total_count": 1, "logits":1, "p_zero":1}
    distr_cls: type = ZeroInflatedNegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor, p_zero: torch.Tensor):
        total_count = F.softplus(total_count).clamp_min(torch.finfo(total_count.dtype).eps)
        p_zero = p_zero.sigmoid().clamp(torch.finfo(p_zero.dtype).eps, 1-torch.finfo(p_zero.dtype).eps)
        return  total_count.squeeze(-1), logits.squeeze(-1), p_zero.squeeze(-1)
    
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        total_count, logits, p_zero = distr_args

        if scale is not None: 
            mu = total_count*torch.exp(logits)  
            logits += ((scale*(1. + mu) -1.)/mu).log()

        return ZeroInflatedNegativeBinomial(total_count, logits, p_zero)
        
    @property
    def event_shape(self) -> Tuple:
        return ()