import math
import json
import torch
from torch.distributions import NegativeBinomial, Poisson
import gpytorch
from gpytorch.priors import GammaPrior
from .distributions import (
    Tweedie, 
    FixedDispersionTweedie
)
from .gp_likelihoods import (
    NegativeBinomialLikelihood, 
    PoissonLikelihood, 
    TweedieLikelihood, 
    FixedDispersionTweedieLikelihood,
    ZeroInflatedPoissonLikelihood,
    ZeroInflatedNegativeBinomialLikelihood,
    HurdleShiftedPoissonLikelihood,
    HurdleShiftedNegativeBinomialLikelihood,
    ZeroInflatedGammaLikelihood
)

from .gp_priors import BetaPrior, LogNormalPrior
from .gp_models import GP, Likelihood


class EarlyStopper():
    """
    This class creates an early stopper for the training of a model.

    Args:
        best_loss (float): the  minimum achieved by the loss.
        streak (int): the amount of consecutive iterations  with no improvement.
        tolerance (int): if the streak reach this value, training is stopped.
        delta (float): the minimum improvement required to set back the streak to 0. 
        best_models (gpytorch.models.ApproximateGP, gpytorch.likelihoods.Lokelihood): the GP and likelihood which gave the best loss.
    """

    def __init__(self, tolerance = 1, delta = 1e-04):
        """
        Initialize the early stopper, specifying the arguments.

        Args:
            tolerance (int): the value of the tolerance to be set.
            delta (int): the value of the delta to be set.
        """

        self.best_loss = math.inf
        self.streak = 0
        self.tolerance = tolerance
        self.delta = delta
        self.best_models = None

    def update(self, new_loss, gp, likelihood):
        """
        See the new value of the loss, and in case update the minimum.

        Args:
            new_loss (float): the last observed value for the loss function.
        """

        if new_loss >= self.best_loss + self.delta:
            self.streak += 1
        else:
            self.best_loss = new_loss
            self.streak = 0
            self.best_models = gp, likelihood

    def is_saturated(self):
        """
        Checks if the streak has reached the tolerance.

        Return:
            bool: this is true in case the training has to be stopped.
        """
        return self.streak >= self.tolerance


class intermittentGP(torch.nn.Module):
    """
    This class creates a Gaussian Process for forecasting intermittent time series.

    Args:
        likelihood (str): the likelihood of the model.
        kernel (Union(None, str)): the kernel used to induce the covariance matrix of the Gaussian Process.
        scaling (Union(None, str)): the kind of scaling to be applied to the data.
        scale_factor (float): the value the data is scaled to.
        num_inducing_points (Union(None, int)): the amount of inducing points got a sparse GP.
        init_inducing_points (Union(None, str)): the scheme for the initializiation of inucing points.
        n_samples (int): the amount of samples generated in predction.
        max_iter (int): number of maximum training iterations.
        min_iter (int): number of minimum training iterations.
        tolerance (int): the toleramce of the early stopping.
        delta (float): the delta of the eraly stopping.
    """

    def __init__(self, 
                 likelihood, 
                 kernel = None, 
                 scaling = None, 
                 scale_factor = 1., 
                 num_inducing_points = None,
                 init_inducing_points = None, 
                 n_samples = 5e04,
                 max_iter = 100, 
                 min_iter = 25, 
                 tolerance = 1, 
                 delta = 1e-04,
                 priors = False):
        
        super().__init__()
        assert likelihood in ['poisson', 'negbin', 'tweedie', 'fixed-dispersion-tweedie', 'zip', 'zinb', 'zig', 'hsp', 'hsnb']
        assert kernel in [None, 'sm', 'linear', 'periodic']
        assert scaling in [None, 'mase', 'mean', 'mean-demand', 'median-demand']

        if priors is not False:
            def dict_to_prior(dict):
                class_name = dict['class_name']
                params = {key: torch.tensor(value) for key, value in dict['params'].items()}
                try:
                    prior = getattr(gpytorch.priors, class_name + 'Prior')
                except:
                    prior = globals()[class_name + 'Prior']
                return prior(**params)     
            with open(priors, 'r') as file:
                priors = {name: dict_to_prior(prior_dict) for name, prior_dict in json.load(file).items()}
        else:
            priors = {}

        self.likelihood = likelihood
        self.kernel = kernel
        self.scaling = scaling
        self.scale_factor = scale_factor
        self.num_inducing_points = num_inducing_points
        self.init_inducing_points = init_inducing_points
        self.n_samples = n_samples
        self.max_iter = max_iter 
        self.min_iter = min_iter
        self.tolerance = tolerance
        self.delta = delta
        self.priors = priors


    def build(self, train_x, train_y):
        """
        Build the GP saving the likelihood and and the process itself (they are GPyTorch instances).

        Args:
            train_x (Union(None, torch.Tensor): the timestamps of training.
            train_y (Union(None, torch.Tensor)): the trainig values.
        """       

        self._gp = GP(train_x, train_y, self.kernel, self.num_inducing_points, self.init_inducing_points, self.priors)

        self._likelihood = Likelihood(self.likelihood, self.priors)
        
        self._early_stopper = EarlyStopper(self.tolerance, self.delta)


    def fit(self, train_x, train_y):
        """
        Trains the variational GP, using early stopping.

        Args:
            train_x (torch.Tensor): the timestamps of training.
            train_y (torch.Tensor): the trainig values.
        """

        if self.scaling is None:
            self._scale = None
        elif self.scaling == 'mase':
            self._scale = torch.mean(torch.abs(train_y[1:] - train_y[:-1]), dtype=torch.float32)
        elif self.scaling == 'mean':
            self._scale = torch.mean(train_y, dtype=torch.float32)
        elif self.scaling == 'mean-demand':
            self._scale = torch.mean(train_y[train_y != 0], dtype=torch.float32)
        elif self.scaling == 'median-demand':
            self._scale = torch.median(train_y[train_y != 0]).to(dtype=torch.float32)
    
        if self._scale and (self._scale == 0 or torch.isnan(self._scale)):
            self._scale = None
        
        if self._scale:
            self._scale /= self.scale_factor
            train_y = train_y/self._scale

        self._early_stopper = EarlyStopper(self.tolerance, self.delta)
        train_longer = self.likelihood == "tweedie" and len(train_y) <= 500

        mll = gpytorch.mlls.VariationalELBO(self._likelihood, self._gp, num_data=train_y.size(0))
        optimizer = torch.optim.Adam(mll.parameters(), lr = 0.1)
        self._stop = None

        for iter in range(self.max_iter * (1+train_longer)):
            self._gp.train(), self._likelihood.train(), mll.train()
            optimizer.zero_grad()

            try:
                pred = self._gp(train_x)
                loss = -mll(pred, train_y)
            except Exception as exc:
                self._stop = str(exc.__class__.__name__)
                break

            loss.backward()
            optimizer.step()

            if iter > (self.min_iter * (1+train_longer)) - self._early_stopper.tolerance - 1:
                self._early_stopper.update(loss.item(), self._gp, self._likelihood)
                if self._early_stopper.is_saturated():
                    self._stop = "Variational loss converged"
                    self._gp, self._likelihood = self._early_stopper.best_models
                    break
        
        if self._stop is  None and iter == (self.max_iter * (1+train_longer))-1:
            self._stop = "Max iterations reached"
        self._iter = iter

    def predict(self, test_x):
        """
        Predicts future values using the appropriate non-gaussian likelihood.

        Args:
            test_x (torch.Tensor): the timestamps of test.

        Reuturns:
            torch.Tensor: the predicted mean.
            torch.tensor: a 2-dimensional tensor of predictive samples.
        """
        
        self._gp.eval(), self._likelihood.eval()
        posterior_gp_test = self._gp(test_x)

        if type(self._likelihood) == TweedieLikelihood:
            mean_pred = torch.nn.functional.softplus(posterior_gp_test.loc.detach())
            samples_pred = Tweedie(mu = torch.nn.functional.softplus(posterior_gp_test.sample(torch.Size([int(self.n_samples)]))), 
                                   phi = self._likelihood.phi, rho = self._likelihood.rho).sample()
        elif type(self._likelihood) == PoissonLikelihood:
            mean_pred = torch.nn.functional.softplus(posterior_gp_test.loc.detach())
            samples_pred = Poisson(torch.nn.functional.softplus(posterior_gp_test.sample(torch.Size([int(self.n_samples)])))).sample()
        elif type(self._likelihood) == NegativeBinomialLikelihood:
            mean_pred = torch.nn.functional.softplus(posterior_gp_test.loc.detach())*(self._likelihood.probs)/(1-self._likelihood.probs)
            samples_pred = NegativeBinomial(total_count = torch.nn.functional.softplus(self._gp(test_x).sample(torch.Size([int(self.n_samples)]))), 
                                            probs = self._likelihood.probs).sample()
        elif type(self._likelihood) == FixedDispersionTweedieLikelihood:
            mean_pred = torch.nn.functional.softplus(posterior_gp_test.loc.detach())
            samples_pred = FixedDispersionTweedie(mu = torch.nn.functional.softplus(posterior_gp_test.sample(torch.Size([int(self.n_samples)]))), 
                                                  rho = self._likelihood.rho).sample()
        elif type(self._likelihood) in [
            ZeroInflatedPoissonLikelihood,
            ZeroInflatedNegativeBinomialLikelihood,
            ZeroInflatedGammaLikelihood,
            HurdleShiftedPoissonLikelihood,
            HurdleShiftedNegativeBinomialLikelihood,
        ]:
            mean_pred = torch.nn.functional.softplus(posterior_gp_test.loc.detach())
            samples_pred = self._likelihood(posterior_gp_test.sample(torch.Size([int(self.n_samples)]))).sample()
            
        else:
        
            raise TypeError('Likelihood not recognised')  
    
        if self._scale:
            mean_pred, samples_pred = mean_pred*self._scale, samples_pred*self._scale

        return mean_pred, samples_pred
