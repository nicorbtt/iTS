from typing import Callable, Dict, Optional, Tuple, List, Union

import torch
from torch import nn
from torch.distributions import (
    AffineTransform,
    Distribution,
    Independent,
    Poisson,
    TransformedDistribution
)

from distributions import Tweedie, FixedDispersionTweedie

import transformers
from transformers import (
    TimeSeriesTransformerConfig, 
    TimeSeriesTransformerForPrediction, 
    TimeSeriesTransformerModel, TimeSeriesMeanDemandScaler,
)
from transformers.time_series_utils import NegativeBinomialOutput, DistributionOutput

from gluonts.torch import distributions

class TimeSeriesMeanDemandScaler(nn.Module):

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  
        demand_sum = (data * observed_indicator)[data > 0].abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator[data > 0].sum(self.dim, keepdim=True)

        scale = demand_sum / torch.clamp(num_observed, min=1)

        if self.default_scale is None:
            batch_sum = demand_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        scale = torch.where(num_observed > 0, scale, default_scale)

        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale
    
    

#class IntermittentTimeSeriesTransformerModel(TimeSeriesTransformerModel):
    #def __init__(self, config: TimeSeriesTransformerConfig):
        #super().__init__(config)

        #if config.scaling == "mean" or config.scaling is True:
            #self.scaler = TimeSeriesMeanScaler(config)
        #elif config.scaling == "std":
            #self.scaler = TimeSeriesStdScaler(config)
        #elif self.scaling == 'mean demand':
            #self.scaler = TimeSeriesMeanDemandScaler(config)
        #else:
            #self.scaler = TimeSeriesNOPScaler(config)

        #if config.num_static_categorical_features > 0:
            #self.embedder = TimeSeriesFeatureEmbedder(
                #cardinalities=config.cardinality,
                #embedding_dims=config.embedding_dimension,
            #)

        # transformer encoder-decoder and mask initializer
        #self.encoder = TimeSeriesTransformerEncoder(config)
        #self.decoder = TimeSeriesTransformerDecoder(config)

        # Initialize weights and apply final processing
        #self.post_init()


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    return -input.log_prob(target)

class TweedieOutput(DistributionOutput, distributions.DistributionOutput):

    args_dim: Dict[str, int] = {"mu": 1, "phi": 1, "rho":1}
    distribution_class: type = Tweedie

    @classmethod
    def domain_map(cls, mu: torch.Tensor, phi: torch.Tensor, rho: torch.Tensor):
        mu = cls.squareplus(mu).clamp_min(torch.finfo(mu.dtype).eps)
        phi = cls.squareplus(phi).clamp_min(torch.finfo(phi.dtype).eps)
        rho = (1+rho.sigmoid()).clamp(1+torch.finfo(rho.dtype).eps, 2-torch.finfo(rho.dtype).eps)
        
        rho = rho.clamp(1+torch.finfo(rho.dtype).eps, 2-torch.finfo(rho.dtype).eps)
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
        
class FixedDispersionTweedieOutput(DistributionOutput, distributions.DistributionOutput):

    args_dim: Dict[str, int] = {"mu": 1, "rho":1}
    distribution_class: type = FixedDispersionTweedie

    @classmethod
    def domain_map(cls, mu: torch.Tensor, rho: torch.Tensor):
        mu = cls.squareplus(mu).clamp_min(torch.finfo(mu.dtype).eps)
        rho = (1+rho.sigmoid()).clamp(1+torch.finfo(rho.dtype).eps, 2-torch.finfo(rho.dtype).eps)
        
        rho = rho.clamp(1+torch.finfo(rho.dtype).eps, 2-torch.finfo(rho.dtype).eps)
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
        

class PoissonOutput(DistributionOutput, distributions.DistributionOutput):

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

        return self._base_distribution((rate))
    
    @property
    def event_shape(self) -> Tuple:
        return ()
    

class IntermittentTimeSeriesTransformerForPrediction(TimeSeriesTransformerForPrediction):
    def __init__(self, config: TimeSeriesTransformerConfig):
        if config.distribution_output in ("tweedie", "fixed_dispersion_tweedie", "poisson", "negative_binomial"):
            distribution_output_aux, config.distribution_output = config.distribution_output, "student_t"
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")
        super().__init__(config)
        self.model = TimeSeriesTransformerModel(config)
        if distribution_output_aux == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        elif distribution_output_aux == "tweedie":
            self.distribution_output = TweedieOutput(dim=config.input_size)
        elif distribution_output_aux == "fixed_dispersion_tweedie":
            self.distribution_output = FixedDispersionTweedieOutput(dim=config.input_size)
        elif distribution_output_aux == "poisson":
            self.distribution_output = PoissonOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")
        
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.d_model)
        self.target_shape = self.distribution_output.event_shape

        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        self.post_init()

        self.config.distribution_output = distribution_output_aux

