import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Poisson, Gamma, Beta, constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

from .distribution_utils import DistributionClass
from ..utils import *


class Tweedie(ExponentialFamily):
    """
    Tweedie distribution class.
    
    The Tweedie distribution is a flexible family of distributions that includes
    Poisson, Gamma, and compound Poisson-Gamma distributions as special cases.
    
    Distributional Parameters
    -------------------------
    mu: torch.Tensor
        Mean parameter (must be positive)
    phi: torch.Tensor
        Dispersion parameter (must be positive)
    rho: torch.Tensor
        Power/shape parameter (must be in [1, 2])
    """
    
    arg_constraints = {
        "mu": constraints.nonnegative,
        "phi": constraints.positive,
        "rho": constraints.interval(1, 2)
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0
    
    @property
    def mean(self):
        return self.mu
        
    @property
    def variance(self):
        return self.phi * torch.pow(self.mu, self.rho)
    
    def __init__(self, mu, phi, rho, validate_args=None):
        self.mu, self.phi, self.rho = broadcast_all(mu, phi, rho)
        if isinstance(mu, (int, float)) and isinstance(phi, (int, float)) and isinstance(rho, (int, float)):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super().__init__(batch_shape, validate_args=validate_args)
    
    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.mu.dtype, device=self.mu.device)
        if self._validate_args:
            self._validate_sample(value)

        def log_prob_nonzero(y, mu, phi, rho):
            
            if y.ndim > 1:
                y = torch.flatten(y)
            if mu.ndim > 1:
                mu = torch.flatten(mu)
            if phi.ndim > 1:
                phi = torch.flatten(phi)
            if rho.ndim > 1:
                rho = torch.flatten(rho)

            def get_alpha(rho):
                return (2 - rho) / (1 - rho)

            def get_jmax(y, phi, rho):
                return torch.pow(y, 2 - rho) / (phi * (2 - rho))

            def get_log_z(y, phi, rho):
                alpha = get_alpha(rho)
                return -alpha * torch.log(y) + alpha * torch.log(rho - 1) - torch.log(2 - rho) - (1 - alpha) * torch.log(phi)
            
            def get_log_W(alpha, j, constant_log_W, pi):
                return (j * (constant_log_W - (1 - alpha) * torch.log(j)) - 
                        torch.log(2 * pi) - 0.5 * torch.log(-alpha) - torch.log(j))
            
            def get_log_W_max(alpha, j, pi):
                return (j * (1 - alpha) - torch.log(2 * pi) - 0.5 * torch.log(-alpha) - torch.log(j))

            pi = torch.tensor(math.pi, device=y.device)
            alpha = get_alpha(rho)
            log_z = get_log_z(y, phi, rho)
        
            if torch.any(torch.isinf(log_z)):
                raise OverflowError("log(z) growing towards infinity")
        
            j_max = get_jmax(y, phi, rho)
            constant_log_W = log_z + (1 - alpha) + alpha * torch.log(-alpha)
            log_W_max = get_log_W_max(alpha, j_max.round(), pi)

            j = max(torch.tensor(1, device=y.device), j_max.max().round())
            log_W = get_log_W(alpha, j, constant_log_W, pi)
            while torch.any((log_W_max - log_W) < 37):
                j += 1
                log_W = get_log_W(alpha, j, constant_log_W, pi)
                if torch.any(torch.isinf(log_W)):
                    break
            j_U = j.item()

            j = max(torch.tensor(1, device=y.device), j_max.min().round())
            log_W = get_log_W(alpha, j, constant_log_W, pi)
            while torch.any(log_W_max - log_W < 37) and j > 1:
                j -= 1
                log_W = get_log_W(alpha, j, constant_log_W, pi)
                if torch.any(torch.isinf(log_W)):
                    break
            j_L = j.item()
        
            j = torch.arange(j_L, j_U + 1, device=y.device)
            j_2dim = torch.tile(j.float(), (log_z.shape[0], 1)).to(torch.float32)
            log_W = j_2dim * log_z[:, None] - torch.special.gammaln(j.float() + 1) - torch.special.gammaln(-alpha[:, None] * j.float())

            max_log_W = torch.max(log_W, axis=1).values
            sum_W = torch.exp(log_W - max_log_W[:, None]).sum(axis=1)

            return max_log_W + torch.log(sum_W) - torch.log(y) + (((y * torch.pow(mu, 1 - rho) / (1 - rho)) - 
                                                            torch.pow(mu, 2 - rho) / (2 - rho)) / phi)
            
        value, mu, phi, rho = broadcast_all(value, self.mu, self.phi, self.rho)

        log_p = torch.full(value.shape, torch.nan, device=value.device)

        zeros = value == 0
        non_zeros = ~zeros
        
        if torch.any(zeros):
            log_p[zeros] = -(torch.pow(mu[zeros], 2 - rho[zeros]) / (phi[zeros] * (2 - rho[zeros])))
        
        if torch.any(non_zeros):
            log_p[non_zeros] = log_prob_nonzero(value[non_zeros], mu[non_zeros], phi[non_zeros], rho[non_zeros])
        
        return log_p
 
    @property
    def poisson_rate(self):
        return torch.pow(self.mu, 2 - self.rho) / (self.phi * (2 - self.rho))

    @property 
    def gamma_concentration(self):
        return (2 - self.rho) / (self.rho - 1)
    
    @property
    def gamma_rate(self):
        return 1 / (self.phi * (self.rho - 1) * torch.pow(self.mu, self.rho - 1))

    def sample(self, sample_shape=torch.Size()):
        
        rate, alpha, beta = self.poisson_rate, self.gamma_concentration, self.gamma_rate
        rate, alpha, beta = broadcast_all(rate, alpha, beta)

        with torch.no_grad():
            samples = Poisson(rate).sample(sample_shape)
            non_zeros = samples > 0

            if torch.any(non_zeros):
                alpha_expanded, beta_expanded = alpha.expand_as(samples), beta.expand_as(samples)
                samples[non_zeros] = Gamma(samples[non_zeros] * alpha_expanded[non_zeros], 
                                          beta_expanded[non_zeros]).sample()

            return samples


class TweedieDistribution(DistributionClass):
    """
    Tweedie distribution class for LightGBMLSS.
    
    The Tweedie distribution is a flexible family that encompasses Poisson, 
    Gamma, and other distributions as special cases.
    
    Distributional Parameters
    -------------------------
    mu: torch.Tensor
        Mean parameter (positive, typically in (0, inf))
    phi: torch.Tensor
        Dispersion parameter (positive, typically in (0, inf))
    rho: torch.Tensor
        Power/shape parameter (in [1, 2])
    
    Source
    ------
    Inspired by gluon-ts implementation
    https://github.com/awslabs/gluonts
    
    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn_mu: str
        Response function for transforming mu to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
    response_fn_phi: str
        Response function for transforming phi to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
    response_fn_rho: str
        Response function for transforming rho to [1, 2]. Currently "sigmoid_rho" is the default.
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    initialize: bool
        Whether to initialize the distributional parameters with unconditional start values.
    """
    
    def __init__(self,
                 stabilization: str = "None",
                 response_fn_mu: str = "softplus",
                 response_fn_phi: str = "softplus",
                 response_fn_rho: str = "sigmoid_rho",
                 loss_fn: str = "nll",
                 initialize: bool = False,
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        # Specify Response Functions for mu
        response_functions_mu = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn_mu in response_functions_mu:
            response_fn_mu = response_functions_mu[response_fn_mu]
        else:
            raise ValueError(
                "Invalid response function for mu. Please choose from 'exp' or 'softplus'.")

        # Specify Response Functions for phi
        response_functions_phi = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn_phi in response_functions_phi:
            response_fn_phi = response_functions_phi[response_fn_phi]
        else:
            raise ValueError(
                "Invalid response function for phi. Please choose from 'exp' or 'softplus'.")

        # Response function for rho: maps to [1, 2]
        def sigmoid_rho_fn(x):
            """Transform from R to [1, 2] interval using sigmoid"""
            return 1.0 + torch.sigmoid(x)
        
        response_functions_rho = {"sigmoid_rho": sigmoid_rho_fn}
        if response_fn_rho in response_functions_rho:
            response_fn_rho = response_functions_rho[response_fn_rho]
        else:
            raise ValueError(
                "Invalid response function for rho. Please choose 'sigmoid_rho'.")

        # Set the parameters specific to the distribution
        distribution = Tweedie
        param_dict = {
            "mu": response_fn_mu,
            "phi": response_fn_phi,
            "rho": response_fn_rho
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn,
                         initialize=initialize,
                         )
