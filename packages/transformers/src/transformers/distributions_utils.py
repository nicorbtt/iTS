from numbers import Number
from typing import Any, Optional
import math

import torch
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions import Poisson, Gamma, Beta, NegativeBinomial

        
class Tweedie(ExponentialFamily):
    
    arg_constraints = {"mu": constraints.nonnegative,
                       "phi": constraints.positive,
                       "rho": constraints.interval(1,2)}
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
        if isinstance(mu, Number) and isinstance(phi, Number) and isinstance(rho, Number):
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

            def get_alpha(rho): return (2-rho)/(1-rho)

            def get_jmax(y, phi, rho): return torch.pow(y, 2 - rho)/(phi*(2 - rho))

            def get_log_z(y, phi, rho): 
                alpha = get_alpha(rho)
                return -alpha*torch.log(y) + alpha*torch.log(rho-1) - torch.log(2 - rho) - (1-alpha)*torch.log(phi)
            
            def get_log_W(alpha, j, constant_log_W, pi):
                return (j * (constant_log_W - (1 - alpha) * torch.log(j)) - 
                        torch.log(2 * pi) - 0.5 * torch.log(-alpha) - torch.log(j))
            
            def get_log_W_max(alpha, j, pi): 
                return (j * (1-alpha) - torch.log(2 * pi) - 0.5 * torch.log(-alpha) - torch.log(j))

            pi = torch.tensor(math.pi)
            alpha = get_alpha(rho)
            log_z = get_log_z(y, phi, rho)
        
            if torch.any(torch.isinf(log_z)):
                raise OverflowError("log(z) growing towards infinity")
        
            j_max = get_jmax(y, phi, rho)
            constant_log_W = log_z +(1-alpha) + alpha*torch.log(-alpha)
            log_W_max = get_log_W_max(alpha, j_max.round(), pi)

            j = max(torch.tensor(1), j_max.max().round())
            log_W = get_log_W(alpha, j, constant_log_W, pi)
            while torch.any((log_W_max - log_W) < 37):
                j += 1
                log_W = get_log_W(alpha, j, constant_log_W, pi)
                if torch.any(torch.isinf(log_W)):
                    break
            j_U = j.item()

            j = max(torch.tensor(1), j_max.min().round()) 
            log_W = get_log_W(alpha, j, constant_log_W, pi)
            while torch.any(log_W_max - log_W < 37)  and j>1:
                j -= 1
                log_W = get_log_W(alpha, j, constant_log_W, pi)
                if torch.any(torch.isinf(log_W)):
                    break
            j_L = j.item()
        
            j = torch.arange(j_L, j_U + 1, device=y.device)  
            j_2dim = torch.tile(j, (log_z.shape[0], 1)).to(torch.float32)
            log_W = j_2dim*log_z[:, None] - torch.special.gammaln(j + 1) - torch.special.gammaln(-alpha[:, None] * j)

            max_log_W = torch.max(log_W, axis=1).values
            sum_W = torch.exp(log_W - max_log_W[:, None]).sum(axis=1)

            return max_log_W + torch.log(sum_W) - torch.log(y) + (((y * torch.pow(mu, 1 -rho)/(1 -rho)) - 
                                                            torch.pow(mu, 2 -rho)/(2 -rho)) / phi)
            
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
    def poisson_rate(self): return torch.pow(self.mu, 2-self.rho)/(self.phi*(2 - self.rho))

    @property 
    def gamma_concentration(self): return (2-self.rho)/(self.rho-1)
    
    @property
    def gamma_rate(self): return 1/(self.phi*(self.rho -1)*torch.pow(self.mu, self.rho -1))

    #PARAMETERIZATION
    
    #lambd = torch.pow(mu, 2-rho)/(phi*(2 - rho))
    #alpha =  (2-rho)/(rho-1)
    #beta =  1/(phi*(rho -1)*torch.pow(mu, rho -1))

    #mu = lambd * alpha / beta
    #phi = (alpha+1)/(torch.pow(beta, alpha/(alpha+1))*torch.pow(lambd*alpha, 1/(alpha+1)))
    #rho = (alpha+2)/(alpha + 1)

    def sample(self, sample_shape=torch.Size()):
        
        rate, alpha, beta = self.poisson_rate, self.gamma_concentration, self.gamma_rate
        rate, alpha, beta = broadcast_all(rate, alpha, beta)

        with torch.no_grad():
            
            samples = Poisson(rate).sample(sample_shape)
            non_zeros = samples > 0

            if torch.any(non_zeros):
                alpha, beta = alpha.expand_as(samples), beta.expand_as(samples)
                samples[non_zeros] = Gamma(samples[non_zeros]*alpha[non_zeros], beta[non_zeros]).sample()

            return samples

class FixedDispersionTweedie(Tweedie):

    arg_constraints = {"mu": constraints.nonnegative,
                       "rho": constraints.interval(1,2)}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, mu, rho, validate_args=None):
        super().__init__(mu, torch.tensor([1.], device=mu.device), rho, validate_args)



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
            
            return torch.bernoulli(torch.broadcast_to(1-p, sample_shape + p.shape))*(1+Poisson(rate).sample(sample_shape))
        
    def cdf(self, value):
        
        raise NotImplementedError
    

class ZeroInflatedNegativeBinomial(ExponentialFamily):

    arg_constraints = {"total_count": constraints.nonnegative,
                       "probs": constraints.interval(0,1),
                       "p_zero": constraints.interval(0,1)}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0
    
    @property
    def mean(self):
        return (1-self.p_zero)*self.total_count*self.probs/(1 - self.probs)
        
    @property
    def variance(self):
        raise NotImplementedError()
    
    def __init__(self, total_count, probs, p_zero, validate_args=None):
        self.total_count, self.probs, self.p_zero = broadcast_all(total_count, probs, p_zero)
        if isinstance(total_count, Number) and isinstance(probs, Number) and isinstance(p_zero, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.total_count.size()
        super().__init__(batch_shape, validate_args=validate_args)


    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.total_count.dtype, device=self.p_zero.device)
        if self._validate_args:
            self._validate_sample(value)
            
        value, total_count, probs, p_zero = broadcast_all(value, self.total_count, self.probs, self.p_zero)
        
        log_p = torch.full(value.shape, torch.nan)

        zeros = value == 0
        non_zeros = ~zeros
        
        if torch.any(zeros):
            log_p[zeros] = torch.log(p_zero[zeros])
        
        if torch.any(non_zeros):
            log_p[non_zeros] = torch.log(1-p_zero[non_zeros]) + NegativeBinomial(total_count[non_zeros], probs[non_zeros]).log_prob(value[non_zeros] - 1)
        
        return log_p
    
    def rsample(self, sample_shape=torch.Size()):
        
        total_count, probs, p_zero = broadcast_all(self.total_count, self.probs, self.p_zero)

        with torch.no_grad():
            
            return torch.bernoulli(torch.broadcast_to(1-p_zero, sample_shape + p_zero.shape))*(1+NegativeBinomial(total_count, probs).sample(sample_shape))