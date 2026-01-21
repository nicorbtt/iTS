import math
import warnings
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import RBFKernel, ScaleKernel, SpectralMixtureKernel, LinearKernel, PeriodicKernel, CosineKernel
from gpytorch.priors import NormalPrior, GammaPrior
from .gp_likelihoods import (
    PoissonLikelihood, 
    TweedieLikelihood, 
    NegativeBinomialLikelihood, 
    FixedDispersionTweedieLikelihood,
    ZeroInflatedPoissonLikelihood,
    ZeroInflatedNegativeBinomialLikelihood,
    ZeroInflatedGammaLikelihood,
    HurdleShiftedPoissonLikelihood,
    HurdleShiftedNegativeBinomialLikelihood
)
from .gp_priors import BetaPrior, LogNormalPrior
from .gp_kernels import SMComponentKernel



class GP(ApproximateGP):

    def __init__(self, train_x, train_y, kernel = None, num_inducing_points = None, init_inducing_points = None, priors = {}):

        if num_inducing_points is not None:

            if init_inducing_points is None:
                init_inducing_points = "log"

            L = len(train_x)
            if init_inducing_points == 'uniform':
                idx = torch.sort(torch.randperm(L)[:num_inducing_points]).values
            elif init_inducing_points == 'log':
                idx = torch.sort(torch.multinomial(torch.log(torch.arange(L)/L+1.), num_inducing_points)).values
            elif init_inducing_points == 'log_alt':
                idx = torch.sort(torch.multinomial(torch.log(torch.arange(L)+1.), num_inducing_points)).values
            elif init_inducing_points == 'linear':
                idx = torch.sort(torch.multinomial(torch.arange(L)+L/2, num_inducing_points))
            else:
                raise ValueError
            train_x = train_x[idx]

        if len(train_x) > 500:
            warnings.warn("Large training size: consider to use inducing points")

        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x,
                                                   variational_distribution, learn_inducing_locations=True)
        super(GP, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(lengthscale_prior=priors.get('lengthscale', None)), 
                                        outputscale_prior=priors.get('outputscale', None))

        assert(kernel in [None, 'sm', 'linear', 'periodic'])
        if kernel == 'sm':
            smk = SpectralMixtureKernel(2)
            try:
                smk.initialize_from_data_empspect(train_x, train_y.to(torch.float32))
            except:
                smk.initialize_from_data(train_x, train_y.to(torch.float32))
            self.covar_module += smk
        elif kernel == 'linear':
            self.covar_module += ScaleKernel(LinearKernel())
        elif kernel == 'periodic':
            pk = ScaleKernel(PeriodicKernel())
            pk.base_kernel._set_period_length(1.)
            pk.base_kernel.raw_period_length.requires_grad = False
            self.covar_module += pk


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    

def Likelihood(likelihood, priors):
    if likelihood == 'poisson':
        return PoissonLikelihood()
    elif likelihood== 'negbin':
        return NegativeBinomialLikelihood(probs_prior = priors.get("probs", None))
    elif likelihood == 'tweedie':
        return TweedieLikelihood(phi_prior = priors.get("phi", None), rho_prior=priors.get("rho", None))
    elif likelihood == 'fixed-dispersion-tweedie':
        return FixedDispersionTweedieLikelihood(rho_prior = priors.get("rho", None))
    elif likelihood == 'zip':
        return ZeroInflatedPoissonLikelihood(theta_prior = priors.get("theta", None))
    elif likelihood == 'zinb':
        return ZeroInflatedNegativeBinomialLikelihood(theta_prior = priors.get("theta", None), tau_prior = priors.get("tau", None))
    elif likelihood == 'zig':
        return ZeroInflatedGammaLikelihood(theta_prior = priors.get("theta", None), tau_prior = priors.get("tau", None))
    elif likelihood == 'hsp':
        return HurdleShiftedPoissonLikelihood(theta_prior = priors.get("theta", None))
    elif likelihood == 'hsnb':
        return HurdleShiftedNegativeBinomialLikelihood(theta_prior = priors.get("theta", None), tau_prior = priors.get("tau", None))

    