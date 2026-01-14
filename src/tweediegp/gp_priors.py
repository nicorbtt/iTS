import torch
from torch.distributions import Beta, Chi2, HalfNormal, LogNormal
from torch.nn import Module as TModule

from gpytorch.priors import Prior
#from gpytorch.utils import _bufferize_attributes

#https://docs.gpytorch.ai/en/latest/_modules/gpytorch/module.html#Module
#https://docs.gpytorch.ai/en/stable/priors.html

class BetaPrior(Prior, Beta):

    def __init__(self, concentration1, concentration0, validate_args=False, transform=None):
        TModule.__init__(self)
        Beta.__init__(self, concentration0=concentration0, concentration1=concentration1, validate_args=validate_args)
        #_bufferize_attributes(self, ("concentration0", "concetration1"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return BetaPrior(self.concentration0.expand(batch_shape), self.concetration1.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(Beta, self).__call__(*args, **kwargs)

class Chi2Prior(Prior, Chi2):
    def __init__(self, df, validate_args=False, transform=None):
        TModule.__init__(self)
        Chi2.__init__(self, df=df, validate_args=validate_args)
        #_bufferize_attributes(self, ("df"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return Chi2Prior(self.df.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(Chi2, self).__call__(*args, **kwargs)
    

class LogNormalPrior(Prior, LogNormal):
    def __init__(self, loc, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        LogNormal.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        #_bufferize_attributes(self, ("loc", "scale"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return LogNormalPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(LogNormal, self).__call__(*args, **kwargs)