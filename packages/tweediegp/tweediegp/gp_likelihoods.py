import math
import torch
from torch import Tensor
from torch.distributions import Poisson, NegativeBinomial, Categorical

from .distributions import (
    Tweedie, 
    FixedDispersionTweedie, 
    ZeroInflatedPoisson, 
    HurdleShiftedPoisson,
    ZeroInflatedNegativeBinomial, 
    HurdleShiftedNegativeBinomial,
    ZeroInflatedGamma
)

import gpytorch
from gpytorch.constraints import Interval, Positive, GreaterThan

from typing import Any, Optional

class PoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):

    r"""
    A  Poisson likelihood model for GP regression.
    Nonnegativity is enforced using a softplus activation.
    It does not have learnable parameters.   
    """

    has_analytic_marginal: bool = True

    def __init__(self):
        super().__init__()
        
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Poisson:
        rates = torch.nn.functional.softplus(function_samples)
        return Poisson(rates)
    
    
class NegativeBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    r"""
    A NegativeBinomial likelihood/noise model for GP regression.
    The GP is used to model the counts parameter; nonnegativity is enforced using a softplus activation. 
    The probabilities of succcess :math:`\p` are a learnable parameter.

    :param batch_shape: The batch shape of the learned probabilities parameter (default: []).
    :param noise_prior: Prior for probabilities parameter :math:`\sigma`.
    :param noise_constraint: Constraint for probabilities parameter :math:`\sigma`.

    :var torch.Tensor probs: :math:`p` parameter (probabilities)
    """
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                probs_prior = None,
                probs_constraint: Optional[Interval] = None,) -> None:
        super().__init__()
        
        
        if probs_constraint is None:
            probs_constraint = Interval(0,1)

        self.raw_probs = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if probs_prior is not None:
            self.register_prior("probs_prior", probs_prior, lambda m: m.probs, lambda m, v: m._set_probs(v))

        self.register_constraint("raw_probs", probs_constraint)
        
    @property
    def probs(self) -> Tensor:
        return self.raw_probs_constraint.transform(self.raw_probs)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        self._set_probs(value)

    def _set_probs(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_probs)
        self.initialize(raw_probs=self.raw_probs_constraint.inverse_transform(value))
        
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> NegativeBinomial:
        trials = torch.nn.functional.softplus(function_samples)
        return NegativeBinomial(trials, probs=torch.clamp(self.probs, 1e-06, 1-1e-06))
    

class TweedieLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                phi_prior = None,
                rho_prior = None,
                rho_constraint: Optional[Interval] = None,
                phi_constraint: Optional[Positive] = None,) -> None:
        super().__init__()
        
        if phi_constraint is None:
            phi_constraint = Positive()

        self.raw_phi = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if phi_prior is not None:
            self.register_prior("phi_prior", phi_prior, lambda m: m.phi, lambda m, v: m._set_phi(v))

        self.register_constraint("raw_phi", phi_constraint)

        if rho_constraint is None:
            rho_constraint = Interval(1,2)

        self.raw_rho = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if rho_prior is not None:
            self.register_prior("rho_prior", rho_prior, lambda m: m.rho-1, lambda m, v: m._set_rho(v+1))

        self.register_constraint("raw_rho", rho_constraint)
        
    @property
    def rho(self) -> Tensor:
        return self.raw_rho_constraint.transform(self.raw_rho)

    @rho.setter
    def rho(self, value: Tensor) -> None:
        self._set_rho(value)

    def _set_rho(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho)
        self.initialize(raw_rho=self.raw_rho_constraint.inverse_transform(value))
        
    @property
    def phi(self) -> Tensor:
        return self.raw_phi_constraint.transform(self.raw_phi)

    @phi.setter
    def phi(self, value: Tensor) -> None:
        self._set_phi(value)

    def _set_phi(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_phi)
        self.initialize(raw_phi=self.raw_phi_constraint.inverse_transform(value))
        
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Tweedie:
        val = torch.nn.functional.softplus(function_samples)
        return Tweedie(mu = val, phi = self.phi, rho=self.rho)
    

class FixedDispersionTweedieLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                rho_prior = None,
                rho_constraint: Optional[Interval] = None,) -> None:
        super().__init__()

        if rho_constraint is None:
            rho_constraint = Interval(1,2)

        self.raw_rho = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if rho_prior is not None:
            self.register_prior("rho_prior", rho_prior, lambda m: m.rho-1, lambda m, v: m._set_rho(v+1))

        self.register_constraint("raw_rho", rho_constraint)
        
    @property
    def rho(self) -> Tensor:
        return self.raw_rho_constraint.transform(self.raw_rho)

    @rho.setter
    def rho(self, value: Tensor) -> None:
        self._set_rho(value)

    def _set_rho(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho)
        self.initialize(raw_rho=self.raw_rho_constraint.inverse_transform(value))
    
        
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Tweedie:
        val = torch.nn.functional.softplus(function_samples)
        return FixedDispersionTweedie(mu = val, rho=self.rho)
    


class ZeroInflatedPoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                theta_prior = None,
                theta_constraint: Optional[Interval] = None,) -> None:
        super().__init__()

        if theta_constraint is None:
            theta_constraint = Positive()

        self.raw_theta = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if theta_prior is not None:
            self.register_prior("theta_prior", theta_prior, lambda m: m.theta, lambda m, v: m._set_theta(v))

        self.register_constraint("raw_theta", theta_constraint)
        
    @property
    def theta(self) -> Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    @theta.setter
    def rho(self, value: Tensor) -> None:
        self._set_theta(value)

    def _set_theta(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_theta)
        self.initialize(raw_theta=self.raw_theta_constraint.inverse_transform(value))
    
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Tweedie:
        mu = torch.nn.functional.softplus(function_samples)
        theta = self.theta
        return ZeroInflatedPoisson(rate = mu + theta, pi = torch.clamp(theta/(mu + theta), 1e-6, 1-1e-6))
    

class HurdleShiftedPoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                theta_prior = None,
                theta_constraint: Optional[Interval] = None,) -> None:
        super().__init__()

        if theta_constraint is None:
            theta_constraint = GreaterThan(1)

        self.raw_theta = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if theta_prior is not None:
            self.register_prior("theta_prior", theta_prior, lambda m: m.theta, lambda m, v: m._set_theta(v))

        self.register_constraint("raw_theta", theta_constraint)
        
    @property
    def theta(self) -> Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    @theta.setter
    def theta(self, value: Tensor) -> None:
        self._set_theta(value)

    def _set_theta(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_theta)
        self.initialize(raw_theta=self.raw_theta_constraint.inverse_transform(value))
    
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Tweedie:
        mu = torch.nn.functional.softplus(function_samples)
        theta = self.theta 
        return HurdleShiftedPoisson(rate = mu+theta-1, pi = torch.clamp(theta/(mu + theta), 1e-6, 1-1e-6))
    


class ZeroInflatedNegativeBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                theta_prior = None,
                tau_prior = None,
                theta_constraint: Optional[Interval] = None,
                tau_constraint: Optional[Positive] = None,) -> None:
        super().__init__()
        
        if theta_constraint is None:
            theta_constraint = Positive()

        self.raw_theta = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if theta_prior is not None:
            self.register_prior("theta_prior", theta_prior, lambda m: m.theta, lambda m, v: m._set_theta(v))

        self.register_constraint("raw_theta", theta_constraint)

        if tau_constraint is None:
            tau_constraint = Interval(0,1)

        self.raw_tau = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if tau_prior is not None:
            self.register_prior("tau_prior", tau_prior, lambda m: m.tau, lambda m, v: m._set_tau(v))

        self.register_constraint("raw_tau", tau_constraint)

    @property
    def theta(self) -> Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    @theta.setter
    def theta(self, value: Tensor) -> None:
        self._set_theta(value)

    def _set_theta(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_theta)
        self.initialize(raw_theta=self.raw_theta_constraint.inverse_transform(value))
        
    @property
    def tau(self) -> Tensor:
        return self.raw_tau_constraint.transform(self.raw_tau)

    @tau.setter
    def tau(self, value: Tensor) -> None:
        self._set_tau(value)

    def _set_tau(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_tau)
        self.initialize(raw_tau=self.raw_tau_constraint.inverse_transform(value))
        
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Tweedie:
        mu = torch.nn.functional.softplus(function_samples)
        theta, tau = self.theta, self.tau
        return ZeroInflatedNegativeBinomial(total_count = (theta + mu)*(1 - tau)/tau, probs = tau, pi = torch.clamp(theta/(theta + mu), 1e-6, 1-1e-6))
    


class HurdleShiftedNegativeBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                theta_prior = None,
                tau_prior = None,
                theta_constraint: Optional[Interval] = None,
                tau_constraint: Optional[Positive] = None,) -> None:
        super().__init__()
        
        if theta_constraint is None:
            theta_constraint = GreaterThan(1)

        self.raw_theta = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if theta_prior is not None:
            self.register_prior("theta_prior", theta_prior, lambda m: m.theta, lambda m, v: m._set_theta(v))

        self.register_constraint("raw_theta", theta_constraint)

        if tau_constraint is None:
            tau_constraint = Interval(0,1)

        self.raw_tau = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if tau_prior is not None:
            self.register_prior("tau_prior", tau_prior, lambda m: m.tau, lambda m, v: m._set_tau(v))

        self.register_constraint("raw_tau", tau_constraint)

    @property
    def theta(self) -> Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    @theta.setter
    def theta(self, value: Tensor) -> None:
        self._set_theta(value)

    def _set_theta(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_theta)
        self.initialize(raw_theta=self.raw_theta_constraint.inverse_transform(value))
        
    @property
    def tau(self) -> Tensor:
        return self.raw_tau_constraint.transform(self.raw_tau)

    @tau.setter
    def tau(self, value: Tensor) -> None:
        self._set_tau(value)

    def _set_tau(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_tau)
        self.initialize(raw_tau=self.raw_tau_constraint.inverse_transform(value))
        
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Tweedie:
        mu = torch.nn.functional.softplus(function_samples)
        theta, tau = self.theta, self.tau
        return HurdleShiftedNegativeBinomial(total_count = (theta + mu - 1)*(1 - tau)/tau, probs = tau, pi = torch.clamp(theta/(theta + mu), 1e-6, 1-1e-6))


class ZeroInflatedGammaLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    
    has_analytic_marginal: bool = True
        
    def __init__(self,
                batch_shape: torch.Size = torch.Size([]),
                theta_prior = None,
                tau_prior = None,
                theta_constraint: Optional[Interval] = None,
                tau_constraint: Optional[Positive] = None,) -> None:
        super().__init__()
        
        if theta_constraint is None:
            theta_constraint = GreaterThan(1)

        self.raw_theta = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if theta_prior is not None:
            self.register_prior("theta_prior", theta_prior, lambda m: m.theta, lambda m, v: m._set_theta(v))

        self.register_constraint("raw_theta", theta_constraint)

        if tau_constraint is None:
            tau_constraint = Interval(0,1)

        self.raw_tau = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if tau_prior is not None:
            self.register_prior("tau_prior", tau_prior, lambda m: m.tau, lambda m, v: m._set_tau(v))

        self.register_constraint("raw_tau", tau_constraint)

    @property
    def theta(self) -> Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    @theta.setter
    def theta(self, value: Tensor) -> None:
        self._set_theta(value)

    def _set_theta(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_theta)
        self.initialize(raw_theta=self.raw_theta_constraint.inverse_transform(value))
        
    @property
    def tau(self) -> Tensor:
        return self.raw_tau_constraint.transform(self.raw_tau)

    @tau.setter
    def tau(self, value: Tensor) -> None:
        self._set_tau(value)

    def _set_tau(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_tau)
        self.initialize(raw_tau=self.raw_tau_constraint.inverse_transform(value))
        
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Tweedie:
        mu = torch.nn.functional.softplus(function_samples)
        theta, tau = self.theta, self.tau
        return ZeroInflatedGamma(concentration = (mu + theta)*tau, rate = tau, pi = torch.clamp(theta/(theta + mu), 1e-6, 1-1e-6))