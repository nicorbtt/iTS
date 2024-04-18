from numbers import Number
from typing import Any, Optional
import math

import torch
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions import Poisson, Gamma

        
class Tweedie(ExponentialFamily):
    
    arg_constraints = {"mu": constraints.nonnegative,
                       "phi": constraints.nonnegative,
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
    
     
    def __get_alpha(self, rho): return (2-rho)/(1-rho)
    
    
    def __get_jmax(self, y, phi, rho): return torch.pow(y, 2 - rho)/(phi*(2 - rho))
    

    def __get_log_z(self, y, phi, rho): 
        alpha = self.__get_alpha(rho)
        return -alpha*torch.log(y) + alpha*torch.log(rho-1) - (2 - rho) - (1-alpha)*torch.log(phi)

    
    def __get_log_W(self, alpha, j, constant_log_W, pi):
        return (j * (constant_log_W - (1 - alpha) * torch.log(j)) - 
                torch.log(2 * pi) - 0.5 * torch.log(-alpha) - torch.log(j))

    
    def __get_log_W_max(self, alpha, j, pi): 
        return (j * (1-alpha) - torch.log(2 * pi) - 0.5 * torch.log(-alpha) - torch.log(j))
        
       
    def log_prob_nonzero(self, y, mu, phi, rho):
        
        if y.ndim > 1:
            y = torch.flatten(y)
        if mu.ndim > 1:
            mu = torch.flatten(mu)
        if phi.ndim > 1:
            phi = torch.flatten(phi)
        if rho.ndim > 1:
            rho = torch.flatten(rho)

        pi = torch.tensor(math.pi)
        alpha = self.__get_alpha(rho)
        log_z = self.__get_log_z(y, phi, rho)
        
        if torch.any(torch.isinf(log_z)):
            idx = torch.arange(len(log_z))[torch.isinf(log_z)]
            raise OverflowError("z growing towards infinity")
        
        j_max = self.__get_jmax(y, phi, rho)
        constant_log_W = torch.max(log_z) +(1-alpha) + alpha*torch.log(-alpha)
        
        j = max(torch.tensor(1), j_max.max())

        log_W_max = self.__get_log_W_max(alpha, j, pi)
        
        log_W = log_W_max
        iter = 0
        while torch.any((log_W_max - log_W)[log_W_max >= log_W] < 37):#QUESTO SERVE PERCHÉ A VOLE log_W CRESCE A DISMISURA :/
            j += 1
            log_W = self.__get_log_W(alpha, j, constant_log_W, pi)
            if torch.any(torch.isinf(log_W)):
                break
        j_high = torch.ceil(j)

        j = max(torch.tensor(1), j_max.min())
        log_W_max = self.__get_log_W_max(alpha, j, pi)
        
        log_W = log_W_max
        while torch.any(log_W_max - log_W < 37)  and torch.all(j > 1):
            j -= 1
            log_W = self.__get_log_W(alpha, j, constant_log_W, pi)
            if torch.any(torch.isinf(log_W)):
                break
        j_low = torch.ceil(j)
        
        j = torch.arange(j_low.item(), j_high.item() + 1)

        w1 = torch.tile(j, (log_z.shape[0], 1)).to(torch.float32)
        w1 *= log_z[:, None]
        w1 -= torch.special.gammaln(j + 1)

        log_W = w1 - torch.special.gammaln(-alpha[:, None] * j)

        log_W_max = torch.max(log_W, axis=1).values
        w = torch.exp(log_W - log_W_max[:, None]).sum(axis=1)

        return log_W_max + torch.log(w) - torch.log(y) + (((y * torch.pow(mu, 1 -rho)/(1 -rho)) - 
                                                         torch.pow(mu, 2 -rho)/(2 -rho)) / phi)
    
    
    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.mu.dtype, device=self.mu.device)
        if self._validate_args:
            self._validate_sample(value)
            
        value, mu, phi, rho = broadcast_all(value, self.mu, self.phi, self.rho)

        #print(f'\nMIN\tmu: {round(torch.min(self.mu).item(),3)}\tphi: {round(torch.min(self.phi).item(),3)}\trho: {round(torch.min(self.rho).item(),3)}')
        #print(f'MAX\tmu: {round(torch.max(self.mu).item(),3)}\tphi: {round(torch.max(self.phi).item(),3)}\trho: {round(torch.max(self.rho).item(),3)}')

        dvc = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(dvc)

        log_p = torch.full(value.shape, torch.nan)

        zeros = value == 0
        non_zeros = ~zeros
        
        if torch.any(zeros):
            log_p[zeros] = -(torch.pow(mu[zeros], 2 - rho[zeros]) / (phi[zeros] * (2 - rho[zeros])))
        
        if torch.any(non_zeros):
            log_p[non_zeros] = self.log_prob_nonzero(value[non_zeros], mu[non_zeros], phi[non_zeros], rho[non_zeros])
        
        return log_p

        
    @property
    def poisson_rate(self): return torch.pow(self.mu, 2-self.rho)/(self.phi*(2 - self.rho))

    @property 
    def gamma_concentration(self): return (2-self.rho)/(self.rho-1)
    
    @property
    def gamma_rate(self): return 1/(self.phi*(self.rho -1)*torch.pow(self.mu, self.rho -1))


    mu, phi, rho = torch.tensor([.5]), torch.tensor([3.4]), torch.tensor([1.5])


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
            
            N = Poisson(rate).sample(sample_shape)
            non_zeros = N > 0
            
            samples = torch.full(N.shape, torch.nan)
            samples[~non_zeros] = 0
            
            if len(alpha) > 1 and len(alpha) == len(non_zeros):    
                alpha = alpha[non_zeros]
            if len(beta) > 1 and len(beta) == len(non_zeros):
                beta = beta[non_zeros]
            N = N[non_zeros]
            
            samples[non_zeros] = Gamma(N*alpha, beta).sample()
            return samples

class FixedDispersionTweedie(Tweedie):

    arg_constraints = {"mu": constraints.nonnegative,
                       "rho": constraints.interval(1,2)}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, mu, rho, validate_args=None):
        super().__init__(mu, torch.tensor([1.]), rho, validate_args)




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
    
