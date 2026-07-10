from numbers import Number

import torch
from torch.distributions import NegativeBinomial, constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

from .distribution_utils import DistributionClass
from ..utils import *


class _HurdleShiftedNegativeBinomialTorch(ExponentialFamily):
    arg_constraints = {
        "total_count": constraints.nonnegative,
        "logits": constraints.real,
        "p_zero": constraints.interval(0, 1),
    }
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
        log_p = torch.full(value.shape, torch.nan, dtype=value.dtype, device=value.device)

        zeros = value == 0
        non_zeros = ~zeros

        if torch.any(zeros):
            log_p[zeros] = torch.log(p_zero[zeros])

        if torch.any(non_zeros):
            log_p[non_zeros] = torch.log(1 - p_zero[non_zeros]) + NegativeBinomial(
                total_count=total_count[non_zeros],
                logits=logits[non_zeros],
            ).log_prob(value[non_zeros] - 1)

        return log_p

    def sample(self, sample_shape=torch.Size()):
        total_count, logits, p_zero = broadcast_all(self.total_count, self.logits, self.p_zero)
        with torch.no_grad():
            return torch.bernoulli(torch.broadcast_to(1 - p_zero, sample_shape + p_zero.shape)) * (
                1 + NegativeBinomial(total_count=total_count, logits=logits).sample(sample_shape)
            )


class HurdleShiftedNegativeBinomial(DistributionClass):
    """
    Hurdle-Shifted Negative Binomial distribution.

    This mirrors GluonTS behavior:
    - P(Y = 0) = p_zero
    - P(Y = k) = (1 - p_zero) * NB(k - 1), for k >= 1
    """

    def __init__(
        self,
        stabilization: str = "None",
        response_fn_total_count: str = "softplus",
        response_fn_logits: str = "identity",
        response_fn_p_zero: str = "sigmoid",
        loss_fn: str = "nll",
        initialize: bool = False,
    ):
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        response_functions_total_count = {"exp": exp_fn, "softplus": softplus_fn, "relu": relu_fn}
        if response_fn_total_count in response_functions_total_count:
            response_fn_total_count = response_functions_total_count[response_fn_total_count]
        else:
            raise ValueError(
                "Invalid response function for total_count. Please choose from 'exp', 'softplus' or 'relu'."
            )

        response_functions_logits = {"identity": identity_fn}
        if response_fn_logits in response_functions_logits:
            response_fn_logits = response_functions_logits[response_fn_logits]
        else:
            raise ValueError("Invalid response function for logits. Please choose 'identity'.")

        response_functions_p_zero = {"sigmoid": sigmoid_fn}
        if response_fn_p_zero in response_functions_p_zero:
            response_fn_p_zero = response_functions_p_zero[response_fn_p_zero]
        else:
            raise ValueError("Invalid response function for p_zero. Please choose 'sigmoid'.")

        distribution = _HurdleShiftedNegativeBinomialTorch
        param_dict = {
            "total_count": response_fn_total_count,
            "logits": response_fn_logits,
            "p_zero": response_fn_p_zero,
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=True,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
            initialize=initialize,
        )