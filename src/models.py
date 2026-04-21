import os
import logging
import numpy as np
import torch
torch.set_num_threads(1)
from gluonts.torch.model.deepar.module import DeepARModel
from gluonts.torch.model.simple_feedforward.module import SimpleFeedForwardModel
from gluonts.torch.model.d_linear.module import DLinearModel
from gluonts.torch.model.patch_tst.module import PatchTSTModel
from gluonts.torch.model.tide.module import TiDEModel
from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    get_lags_for_frequency,
)
from gluonts.torch.scaler import MASEScaler, MeanDemandScaler
from gluonts.torch.distributions import (
    PoissonOutput, 
    NegativeBinomialOutput, 
    TweedieOutput,
    FixedDispersionTweedieOutput,
    HurdleShiftedPoissonOutput,
    HurdleShiftedNegativeBinomialOutput,
)
from transformers import (
    TimeSeriesTransformerConfig, 
    TimeSeriesTransformerForPrediction, 
    InformerConfig,
    InformerForPrediction,
    AutoformerConfig,
    AutoformerForPrediction
)
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
# from lightgbmlss.distributions.Tweedie import Tweedie
# from lightgbmlss.distributions.ZeroInflatedNegativeBinomial import ZeroInflatedNegativeBinomial

torch.set_num_threads(1)

from tweediegp.intermittent_gp import intermittentGP

### Configuration dictionary
class ModelConfigBuilder:
    
    def __init__(self, model, distribution_head, scaling):
        assert model in ["deepAR", "transformer", "informer", "autoformer", "patchTST", "tide", "dlinear", "feedforward", "lightgbm"]
        assert distribution_head in ["poisson","negbin", "tweedie", "hsp", "hsnb"]
        assert scaling in ["mase", "mean", "mean-demand", None]
        self.model = model
        self.distribution_head = distribution_head
        self.scaling = scaling
        self._TUNABLE_PARAMS_DEEPAR = {
            'context_length', 'prediction_length','embedding_dimension','num_layers',
            'hidden_size','dropout_rate','num_parallel_samples'
        }
        self._TUNABLE_PARAMS_TRANSFORMER = {
            'context_length', 'prediction_length', 'embedding_dimension','d_model', 
            'encoder_layers', 'decoder_layers', 'encoder_attention_heads', 'decoder_attention_heads', 
            'encoder_ffn_dim', 'decoder_ffn_dim', 'activation_function', 'dropout', 'encoder_layerdrop', 
            'decoder_layerdrop', 'attention_dropout', 'activation_dropout', 'num_parallel_samples', 
            'init_std', 'use_cache'     
        }
        self._TUNABLE_PARAMS_INFORMER = {
            'context_length', 'prediction_length', 'embedding_dimension','d_model', 
            'encoder_layers', 'decoder_layers', 'encoder_attention_heads', 'decoder_attention_heads', 
            'encoder_ffn_dim', 'decoder_ffn_dim', 'activation_function', 'dropout', 'encoder_layerdrop', 
            'decoder_layerdrop', 'attention_dropout', 'activation_dropout', 'num_parallel_samples', 
            'init_std', 'use_cache', 'attention_type', 'sampling_factor', 'distil'
        }
        self._TUNABLE_PARAMS_AUTOFORMER = {
            'context_length', 'prediction_length', 'embedding_dimension','d_model', 
            'encoder_layers', 'decoder_layers', 'encoder_attention_heads', 'decoder_attention_heads', 
            'encoder_ffn_dim', 'decoder_ffn_dim', 'activation_function', 'dropout', 'encoder_layerdrop', 
            'decoder_layerdrop', 'attention_dropout', 'activation_dropout', 'num_parallel_samples', 
            'init_std', 'use_cache', 'label_length', 'moving_average', 'autocorrelation_factor'
        }
        self._TUNABLE_PARAMS_FEEDFORWARD = {
            "prediction_length", "context_length", "hidden_dimensions"
        }
        self._TUNABLE_PARAMS_DLINEAR = {
            "prediction_length", "context_length", "hidden_dimension", "kernel_size"
        }
        self._TUNABLE_PARAMS_PATCHTST = {
            'prediction_length', 'context_length', 'patch_len', 'stride', 'padding_patch',
            'd_model', 'nhead', 'dim_feedforward', 'num_encoder_layers',
            'dropout', 'activation', 'norm_first'
        }
        self._TUNABLE_PARAMS_TIDE = {
            'prediction_length', 'context_length', 'num_feat_dynamic_proj',
            'feat_proj_hidden_dim', 'encoder_hidden_dim', 'decoder_hidden_dim',
            'temporal_hidden_dim', 'distr_hidden_dim', 'decoder_output_dim',
            'dropout_rate', 'num_layers_encoder', 'num_layers_decoder',
            'layer_norm', 'embedding_dimension'
        }
        self._TUNABLE_PARAMS_LIGHTGBM = {
            'context_length', 'prediction_length', 'dist'
        }
        self.params = None

    def build(self, data_info, **kwargs) -> None:
        def _check(key, default_value):
            return kwargs[key] if key in kwargs else default_value
        
        lags_sequence = get_lags_for_frequency(data_info['freq'] if data_info['freq'] != "M" else "ME") if not 'lag_sequence' in kwargs else []
        self.time_features = time_features_from_frequency_str(data_info['freq'] if data_info['freq'] != "M" else "ME") if not 'time_features' in kwargs else []
        
        if self.model == "deepAR":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_DEEPAR):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_DEEPAR}")
            self.params = {
                'freq' : data_info['freq'],
                'context_length' : _check('context_length', data_info['h']*data_info['w']),
                'prediction_length' : _check('prediction_length', data_info['h']),
                'num_feat_dynamic_real' : 0,
                'num_feat_static_real' : 0,
                'num_feat_static_cat' : 1,
                'cardinality' : [data_info['N']],
                'embedding_dimension' : _check('embedding_dimension', [3]),
                'num_layers' : _check('num_layers', 2),
                'hidden_size' : _check('hidden_size', 40),
                'dropout_rate' : _check('dropout_rate', 0.1),
                'distr_output' : {
                        'poisson' : PoissonOutput(),
                        'negbin' : NegativeBinomialOutput(),
                        'tweedie' : TweedieOutput(),
                        'tweedie-fix' : FixedDispersionTweedieOutput(),
                        'hsp' : HurdleShiftedPoissonOutput(),
                        'hsnb' : HurdleShiftedNegativeBinomialOutput(),
                        # 'quantile' : QuantileOutput([0.5, 0.8, 0.9, 0.95, .99]),
                        # 'iqn' : ImplicitQuantileNetworkOutput(),
                        # 'isqf' : ISQFOutput(num_pieces=6, qk_x=[0.5, 0.8, 0.9, 0.95, .99]),
                    }[self.distribution_head],
                'lags_seq' : lags_sequence,
                'scaling' : {
                        'mase' : 'MASE',
                        'mean' : 'mean',
                        'mean-demand' : 'mean demand',
                    }[self.scaling] if self.scaling else False,
                'default_scale' : None,
                'num_parallel_samples' : _check('num_parallel_samples', 100)
            }

        if self.model == "transformer":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_TRANSFORMER):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_TRANSFORMER}")
            self.params = TimeSeriesTransformerConfig(
                prediction_length = _check('prediction_length', data_info['h']),
                context_length = _check('context_length', data_info['h']*data_info['w']),
                distribution_output = {
                        'poisson' : 'poisson',
                        'negbin' : 'negative_binomial',
                        'tweedie' : 'tweedie',
                        'tweedie-fix' : 'fixed_dispersion_tweedie',
                        'tweedie-priors' : 'tweedie_with_priors',
                        'hsp' : 'hurdle_shifted_poisson',
                        'hsnb' : 'hurdle_shifted_negative_binomial',
                    }[self.distribution_head],
                loss = "nll",
                input_size = 1,
                scaling = {
                        'mase' : 'MASE',
                        'mean' : 'mean',
                        'mean-demand' : 'mean demand'
                    }[self.scaling] if self.scaling else None,
                lags_sequence = lags_sequence,
                num_time_features = len(self.time_features) + 1,  # +1 is Age
                num_dynamic_real_features = 0,
                num_static_categorical_features = 1,
                num_static_real_features = 0,
                cardinality = [data_info['N']],
                embedding_dimension = _check('embedding_dimension', [3]),
                
                # architecture params
                d_model = _check('d_model', 32),                                    # Dimensionality of the transformer layers                   
                encoder_layers = _check('encoder_layers', 4),                       # Number of encoder layers              
                decoder_layers = _check('decoder_layers', 4),                       # Number of decoder layers                                          
                encoder_attention_heads = _check('encoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer encoder         
                decoder_attention_heads = _check('decoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer decoder   
                encoder_ffn_dim = _check('encoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in encoder                
                decoder_ffn_dim = _check('decoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in decoder 
                activation_function = _check('activation_function', "gelu"),        # The non-linear activation function (function or string) in the encoder and decoder
                dropout = _check('dropout', 0.1),                                   # The dropout probability for all fully connected layers in the encoder, and decoder
                encoder_layerdrop = _check('encoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each encoder layer  
                decoder_layerdrop = _check('decoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each decoder layer
                attention_dropout = _check('attention_dropout', 0.1),               # The dropout probability for the attention probabilities
                activation_dropout = _check('activation_dropout', 0.1),             # The dropout probability used between the two layers of the feed-forward networks
                num_parallel_samples = _check('num_parallel_samples', 200),         # The number of samples to generate in parallel for each time step of inference
                init_std = _check('init_std', 0.02),                                # The standard deviation of the truncated normal weight initialization distribution
                use_cache = _check('use_cache', True),                              # Whether to use the past key/values attentions (if applicable to the model) to speed up decoding
            )

        if self.model == "informer":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_INFORMER):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_INFORMER}")
            self.params = InformerConfig(
                prediction_length = _check('prediction_length', data_info['h']),
                context_length = _check('context_length', data_info['h']*data_info['w']),
                distribution_output = {
                        'poisson' : 'poisson',
                        'negbin' : 'negative_binomial',
                        'tweedie' : 'tweedie',
                        'tweedie-fix' : 'fixed_dispersion_tweedie',
                        'tweedie-priors' : 'tweedie_with_priors',
                        'hsp' : 'hurdle_shifted_poisson',
                        'hsnb' : 'hurdle_shifted_negative_binomial'
                    }[self.distribution_head],
                loss = "nll",
                input_size = 1,
                scaling = {
                        'mase' : 'MASE',
                        'mean' : 'mean',
                        'mean-demand' : 'mean demand'
                    }[self.scaling] if self.scaling else None,
                lags_sequence = lags_sequence,
                num_time_features = len(self.time_features) + 1,  # +1 is Age
                num_dynamic_real_features = 0,
                num_static_categorical_features = 1,
                num_static_real_features = 0,
                cardinality = [data_info['N']],
                embedding_dimension = _check('embedding_dimension', [3]),
                
                # architecture params
                d_model = _check('d_model', 32),                                    # Dimensionality of the transformer layers                   
                encoder_layers = _check('encoder_layers', 4),                       # Number of encoder layers              
                decoder_layers = _check('decoder_layers', 4),                       # Number of decoder layers                                          
                encoder_attention_heads = _check('encoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer encoder         
                decoder_attention_heads = _check('decoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer decoder   
                encoder_ffn_dim = _check('encoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in encoder                
                decoder_ffn_dim = _check('decoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in decoder 
                activation_function = _check('activation_function', "gelu"),        # The non-linear activation function (function or string) in the encoder and decoder
                dropout = _check('dropout', 0.1),                                   # The dropout probability for all fully connected layers in the encoder, and decoder
                encoder_layerdrop = _check('encoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each encoder layer  
                decoder_layerdrop = _check('decoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each decoder layer
                attention_dropout = _check('attention_dropout', 0.1),               # The dropout probability for the attention probabilities
                activation_dropout = _check('activation_dropout', 0.1),             # The dropout probability used between the two layers of the feed-forward networks
                num_parallel_samples = _check('num_parallel_samples', 200),         # The number of samples to generate in parallel for each time step of inference
                init_std = _check('init_std', 0.02),                                # The standard deviation of the truncated normal weight initialization distribution
                use_cache = _check('use_cache', True),                              # Whether to use the past key/values attentions (if applicable to the model) to speed up decoding
                attention_type = _check('attention_type', 'prob'),                   
                sampling_factor = _check('sampling_factor', 5),
                distil = _check('distil', True)    
            )

        if self.model == "autoformer":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_AUTOFORMER):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_AUTOFORMER}")
            self.params = AutoformerConfig(
                prediction_length = _check('prediction_length', data_info['h']),
                context_length = _check('context_length', data_info['h']*data_info['w']),
                distribution_output = {
                        'poisson' : 'poisson',
                        'negbin' : 'negative_binomial',
                        'tweedie' : 'tweedie',
                        'tweedie-fix' : 'fixed_dispersion_tweedie',
                        'tweedie-priors' : 'tweedie_with_priors',
                        'hsp' : 'hurdle_shifted_poisson',
                        'hsnb' : 'hurdle_shifted_negative_binomial'
                    }[self.distribution_head],
                loss = "nll",
                input_size = 1,
                scaling = {
                        'mase' : 'MASE',
                        'mean' : 'mean',
                        'mean-demand' : 'mean demand'
                    }[self.scaling] if self.scaling else None,
                lags_sequence = lags_sequence,
                num_time_features = len(self.time_features) + 1,  # +1 is Age
                num_dynamic_real_features = 0,
                num_static_categorical_features = 1,
                num_static_real_features = 0,
                cardinality = [data_info['N']],
                embedding_dimension = _check('embedding_dimension', [3]),
                
                # architecture params
                d_model = _check('d_model', 32),                                    # Dimensionality of the transformer layers                   
                encoder_layers = _check('encoder_layers', 4),                       # Number of encoder layers              
                decoder_layers = _check('decoder_layers', 4),                       # Number of decoder layers                                          
                encoder_attention_heads = _check('encoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer encoder         
                decoder_attention_heads = _check('decoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer decoder   
                encoder_ffn_dim = _check('encoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in encoder                
                decoder_ffn_dim = _check('decoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in decoder 
                activation_function = _check('activation_function', "gelu"),        # The non-linear activation function (function or string) in the encoder and decoder
                dropout = _check('dropout', 0.1),                                   # The dropout probability for all fully connected layers in the encoder, and decoder
                encoder_layerdrop = _check('encoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each encoder layer  
                decoder_layerdrop = _check('decoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each decoder layer
                attention_dropout = _check('attention_dropout', 0.1),               # The dropout probability for the attention probabilities
                activation_dropout = _check('activation_dropout', 0.1),             # The dropout probability used between the two layers of the feed-forward networks
                num_parallel_samples = _check('num_parallel_samples', 200),         # The number of samples to generate in parallel for each time step of inference
                init_std = _check('init_std', 0.02),                                # The standard deviation of the truncated normal weight initialization distribution
                use_cache = _check('use_cache', True),                              # Whether to use the past key/values attentions (if applicable to the model) to speed up decoding
                label_length = _check('label_length', 10),
                moving_average = _check('moving_average', 25),
                autocorrelation_factor = _check('autocorrelation_factor', 3)
            )

        if self.model == "feedforward":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_FEEDFORWARD):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_FEEDFORWARD}")
            self.params = {
                'context_length' : _check('context_length', data_info['h']*data_info['w']),
                'prediction_length' : _check('prediction_length', data_info['h']),
                'distr_output' : {
                        'poisson' : PoissonOutput(),
                        'negbin' : NegativeBinomialOutput(),
                        'tweedie' : TweedieOutput(),
                        'hsp' : HurdleShiftedPoissonOutput(),
                        'hsnb' : HurdleShiftedNegativeBinomialOutput(),
                    }[self.distribution_head],
                'hidden_dimensions' : _check('hidden_dimensions', [32, 32, 32, 32, 32]),
                'scale' : {
                        'mase' : 'mase',
                        'mean' : 'mean',
                        'mean-demand' : 'mean-demand'
                    }[self.scaling] if self.scaling else False
            }

        if self.model == "patchTST":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_PATCHTST):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_PATCHTST}")
            self.params = {
                'prediction_length' : _check('prediction_length', data_info['h']),
                'context_length' : _check('context_length', data_info['h']*data_info['w']),
                'patch_len' : _check('patch_len', data_info['h']),
                'stride' : _check('stride', data_info['h'] // 2),
                'padding_patch' : _check('padding_patch', 'end'),
                'd_model' : _check('d_model', 32),
                'nhead' : _check('nhead', 2),
                'dim_feedforward' : _check('dim_feedforward', 32),
                'num_feat_dynamic_real' : 0,
                'num_encoder_layers' : _check('num_encoder_layers', 2),
                'dropout' : _check('dropout', 0.1),
                'activation' : _check('activation', 'relu'),
                'norm_first' : _check('norm_first', False),
                'scaling' : {
                        'mase' : 'MASE',
                        'mean' : 'mean',
                        'mean-demand' : 'mean-demand',
                    }[self.scaling] if self.scaling else None,
                'distr_output' : {
                        'poisson' : PoissonOutput(),
                        'negbin' : NegativeBinomialOutput(),
                        'tweedie' : TweedieOutput(),
                        'tweedie-fix' : FixedDispersionTweedieOutput(),
                        'hsp' : HurdleShiftedPoissonOutput(),
                        'hsnb' : HurdleShiftedNegativeBinomialOutput(),
                    }[self.distribution_head],
            }

        if self.model == "tide":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_TIDE):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_TIDE}")
            self.params = {
                'context_length' : _check('context_length', data_info['h']*data_info['w']),
                'prediction_length' : _check('prediction_length', data_info['h']),
                'num_feat_dynamic_real' : 0,
                'num_feat_dynamic_proj' : _check('num_feat_dynamic_proj', 2),
                'num_feat_static_real' : 0,
                'num_feat_static_cat' : 1,
                'cardinality' : [data_info['N']],
                'embedding_dimension' : _check('embedding_dimension', [3]),
                'feat_proj_hidden_dim' : _check('feat_proj_hidden_dim', 4),
                'encoder_hidden_dim' : _check('encoder_hidden_dim', 32),
                'decoder_hidden_dim' : _check('decoder_hidden_dim', 32),
                'temporal_hidden_dim' : _check('temporal_hidden_dim', 32),
                'distr_hidden_dim' : _check('distr_hidden_dim', 32),
                'decoder_output_dim' : _check('decoder_output_dim', 16),
                'dropout_rate' : _check('dropout_rate', 0.1),
                'num_layers_encoder' : _check('num_layers_encoder', 2),
                'num_layers_decoder' : _check('num_layers_decoder', 2),
                'layer_norm' : _check('layer_norm', False),
                'distr_output' : {
                        'poisson' : PoissonOutput(),
                        'negbin' : NegativeBinomialOutput(),
                        'tweedie' : TweedieOutput(),
                        'hsp' : HurdleShiftedPoissonOutput(),
                        'hsnb' : HurdleShiftedNegativeBinomialOutput(),
                    }[self.distribution_head],
                'scaling' : {
                        'mase' : None,
                        'mean' : 'mean',
                        'mean-demand' : 'mean-demand',
                    }[self.scaling] if self.scaling else None,
            }

        if self.model == "dlinear":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_DLINEAR):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_DLINEAR}")
            self.params = {
                'context_length' : _check('context_length', data_info['h']*data_info['w']),
                'prediction_length' : _check('prediction_length', data_info['h']),
                'distr_output' : {
                        'poisson' : PoissonOutput(),
                        'negbin' : NegativeBinomialOutput(),
                        'tweedie' : TweedieOutput(),
                        'hsp' : HurdleShiftedPoissonOutput(),
                        'hsnb' : HurdleShiftedNegativeBinomialOutput(),
                    }[self.distribution_head],
                'hidden_dimension' : _check('hidden_dimension', 32),
                'kernel_size' : _check('kernel_size', 25),
                'scaling' : {
                        'mase' : 'mase',
                        'mean' : 'mean',
                        'mean-demand' : 'mean-demand'
                    }[self.scaling] if self.scaling else False
            }

        if self.model == "lightgbm":
            if not set(kwargs.keys()).issubset(self._TUNABLE_PARAMS_LIGHTGBM):
                raise ValueError(f"Non-tunable parameter found \nThe set of possible parameter is {self._TUNABLE_PARAMS_LIGHTGBM}")
            if self.scaling is not None:
                raise ValueError("Scaling is not supported for LightGBM distributional models.")
            self.params = {
                'dist' : self.distribution_head,
                'prediction_length' : _check('prediction_length', data_info['h']),
                'context_length' : _check('context_length', data_info['h']*data_info['w']),
                'lags_seq' : lags_sequence,
                'num_feat_dynamic_real' : 0,
                'num_feat_static_real' : 0,
                'num_feat_static_cat' : 1,
            }

    ### Create Model
    def get_model(self):
        if self.model == "deepAR" : 
            tmp = self.params['num_feat_dynamic_real'] + len(self.time_features) + 1
            return(DeepARModel(**({**self.params, 'num_feat_dynamic_real': tmp, 'num_feat_static_real':1})))
        if self.model == "transformer" : 
            return(TimeSeriesTransformerForPrediction(self.params))
        if self.model == 'informer':
            return(InformerForPrediction(self.params))
        if self.model == 'autoformer':
            return(AutoformerForPrediction(self.params))
        if self.model == 'patchTST':
            return(PatchTSTModel(**self.params))
        if self.model == 'tide':
            tmp = self.params['num_feat_dynamic_real'] + len(self.time_features) + 1
            return(TiDEModel(**({**self.params, 'num_feat_dynamic_real': tmp, 'num_feat_static_real':1})))
        if self.model == "feedforward":
            return(SimpleFeedForwardModel(**self.params))
        if self.model == "dlinear":
            return(DLinearModel(**self.params))
        if self.model == "lightgbm":
            return(LightGBMLSS(dist = {
                        'negbin' : NegativeBinomial(response_fn_total_count='softplus'),
                        # 'tweedie' : Tweedie(),
                        # 'hsnb' : HurdleShiftedNegativeBinomial()
                    }[self.params['dist']]))
    
    ### Export config
    def export_config(self):
        if not self.params: raise Exception("Configuration not yet built")
        if self.model == "deepAR":
            return {key: self.params[key] for key in self._TUNABLE_PARAMS_DEEPAR}
        elif self.model == "transformer":
            return {key: self.params.__dict__[key] for key in self._TUNABLE_PARAMS_TRANSFORMER}
        elif self.model == 'informer':
            return {key: self.params.__dict__[key] for key in self._TUNABLE_PARAMS_INFORMER}
        elif self.model == 'autoformer':
            return {key: self.params.__dict__[key] for key in self._TUNABLE_PARAMS_AUTOFORMER}
        elif self.model == "feedforward":
            return {key: self.params[key] for key in self._TUNABLE_PARAMS_FEEDFORWARD}
        elif self.model == "dlinear":
            return {key: self.params[key] for key in self._TUNABLE_PARAMS_DLINEAR}
        elif self.model == 'patchTST':
            return {key: self.params[key] for key in self._TUNABLE_PARAMS_PATCHTST}
        elif self.model == 'tide':
            return {key: self.params[key] for key in self._TUNABLE_PARAMS_TIDE}
        elif self.model == 'lightgbm':
            return {key: self.params[key] for key in self._TUNABLE_PARAMS_LIGHTGBM}

### Forward step
def forward(model, batch, device, config):
    loss = None
    # def _ensure_channel_dim(tensor):
    #     return tensor.unsqueeze(-1) if tensor.dim() == 2 else tensor
    if isinstance(model, PatchTSTModel):
        loss = model.loss(
            past_target = batch['past_values'].to(device),
            future_target = batch['future_values'].to(device),
            past_observed_values = batch['past_observed_mask'].to(device),
            future_observed_values = batch['future_observed_mask'].to(device),
        ).mean()
    if isinstance(model, TiDEModel):
        loss = model.loss(
            feat_static_real=torch.zeros((batch['past_values'].shape[0], 1), device=device),
            feat_static_cat=batch["static_categorical_features"].to(device),
            past_time_feat=batch["past_time_features"].to(device),
            past_target=batch['past_values'].to(device),
            past_observed_values=batch['past_observed_mask'].to(device),
            future_time_feat=batch['future_time_features'].to(device),
            future_target=batch['future_values'].to(device),
            future_observed_values=batch['future_observed_mask'].to(device),
        ).mean()
    if isinstance(model, TimeSeriesTransformerForPrediction):
        loss = model(
            static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
            static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        ).loss
    if isinstance(model, InformerForPrediction):
        loss = model(
            static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
            static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        ).loss
    if isinstance(model, AutoformerForPrediction):
        loss = model(
            static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
            static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        ).loss
    if isinstance(model, DeepARModel):
        loss = model.loss(
            feat_static_cat=batch["static_categorical_features"].to(device),
            feat_static_real=torch.zeros((batch['past_values'].shape[0],1), device=device),
            past_time_feat=batch["past_time_features"].to(device),
            past_target=batch['past_values'].to(device),
            future_time_feat=batch['future_time_features'].to(device),
            future_target=batch['future_values'].to(device),
            past_observed_values=batch['past_observed_mask'].to(device),
            future_observed_values=batch["future_observed_mask"].to(device),
        ).mean()
    if isinstance(model, SimpleFeedForwardModel):
        loss = model.loss(
            past_target = batch['past_values'].to(device),
            future_target = batch['future_values'].to(device),
            future_observed_values = batch['future_observed_mask'].to(device),
        ).mean()
    if isinstance(model, DLinearModel):
        loss = model.loss(
            past_target = batch['past_values'].to(device),
            future_target = batch['future_values'].to(device),
            past_observed_values = batch['past_observed_mask'].to(device),
            future_observed_values = batch['future_observed_mask'].to(device),
        ).mean()
    if isinstance(model, LightGBMLSS):
        X, y = batch
        params_pred = model.predict(X, pred_type="parameters").values  
        _, nll = model.dist.get_params_loss(params_pred, 
                                            torch.tensor(y).reshape(-1, 1),
                                            start_values = [None]*params_pred.shape[1], 
                                            requires_grad=False)
        loss = (nll / len(y)).item()
    return(loss)

# # # 2. Prepare target and start_values
# target = torch.tensor(y.reshape(-1, 1))
# start_values = [0.5] * lgblss.dist.n_dist_param  # Use model.start_values or distribution default

# # # 3. Compute NLL
# _, nll = lgblss.dist.get_params_loss(params_pred, target, start_values, requires_grad=False)
#     return(loss)

### Generate forecasts
def predict(model, batch, device, config):
    predictions = None
    if isinstance(model, PatchTSTModel):
        distr_args, loc, scale = model(
            past_target = batch['past_values'].to(device),
            past_observed_values = batch['past_observed_mask'].to(device),
        )
        predictions = model.distr_output.distribution(
            distr_args, loc=loc, scale=scale
        ).sample(torch.Size([10000])).detach().cpu().numpy().swapaxes(0,1)
    if isinstance(model, TiDEModel):
        distr_args, loc, scale = model(
            feat_static_real=torch.zeros((batch['past_values'].shape[0], 1), device=device),
            feat_static_cat=batch["static_categorical_features"].to(device),
            past_time_feat=batch["past_time_features"].to(device),
            past_target=batch['past_values'].to(device),
            past_observed_values=batch['past_observed_mask'].to(device),
            future_time_feat=batch['future_time_features'].to(device),
        )
        predictions = model.distr_output.distribution(
            distr_args, loc=loc, scale=scale
        ).sample(torch.Size([10000])).detach().cpu().numpy().swapaxes(0,1)
    if isinstance(model, TimeSeriesTransformerForPrediction):
        predictions = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
            static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        ).sequences.cpu().numpy()
    if isinstance(model, InformerForPrediction):
        predictions = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
            static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        ).sequences.cpu().numpy()
    if isinstance(model, AutoformerForPrediction):
        predictions = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
            static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        ).sequences.cpu().numpy()
    if isinstance(model, DeepARModel):
        predictions = model.forward(
            feat_static_cat = batch["static_categorical_features"].to(device),
            feat_static_real= torch.zeros((batch['past_values'].shape[0],1), device=device),
            past_time_feat = batch["past_time_features"].to(device),
            past_target = batch['past_values'].to(device),
            future_time_feat = batch['future_time_features'].to(device),
            past_observed_values = batch['past_observed_mask'].to(device),
            num_parallel_samples = config['num_parallel_samples']
        ).detach().cpu().numpy()
    if isinstance(model, SimpleFeedForwardModel):
        distr_args, loc, scale = model(
            batch['past_values'].to(device)
            )
        predictions = model.distr_output.distribution(
            distr_args, loc=loc, scale=scale
            ).sample(torch.Size([10000])).detach().cpu().numpy().swapaxes(0,1)
    if isinstance(model, DLinearModel):
        distr_args, loc, scale = model(
            batch['past_values'].to(device),
            past_observed_values = batch['past_observed_mask'].to(device)
            )
        predictions = model.distr_output.distribution(
            distr_args, loc=loc, scale=scale
            ).sample(torch.Size([10000])).detach().cpu().numpy().swapaxes(0,1)
    if isinstance(model, LightGBMLSS):
        predictions = model.predict(batch, pred_type="samples", n_samples=10000).values
    return(predictions)


class EarlyStop():
    def __init__(self, logger, patience=20, min_delta = 0.001) -> None:
        self.best_val_loss = np.inf
        self.best_model = None
        self.current_patience = 0
        self.logger = logger
        self.patience = patience
        self.min_delta = min_delta
        self.stop = False

    def update(self, model, epoch, new_val_loss):
        if new_val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = new_val_loss
            self.best_model = model.state_dict()
            self.logger.log_earlystop_newbest(self.best_val_loss)
            self.current_patience = 0
        else:
            self.current_patience = self.current_patience + 1
            if self.current_patience == self.patience:
                self.logger.log_earlystop_stop(epoch, self.best_val_loss)
                self.stop = True

class ParamSampler():

    def __init__(self, param_space=None):
        self.param_space = param_space or {
            "learning_rate": ("float", {"low": 1e-4, "high": 0.1, "log": True}),
            "max_depth": ("int", {"low": 1, "high": 10, "log": False}),
            "num_leaves": ("int", {"low": 2, "high": 200, "log": True}),
            "min_data_in_leaf": ("int", {"low": 1, "high": 64, "log": True}),
            "min_gain_to_split": ("float", {"low": 1e-8, "high": 40.0, "log": False}),
            "min_sum_hessian_in_leaf": ("float", {"low": 1e-8, "high": 40.0, "log": True}),
            "subsample": ("float", {"low": 0.7, "high": 1.0, "log": False}),
            "feature_fraction": ("float", {"low": 0.4, "high": 1.0, "log": False}),
            "boosting": ("categorical", ["gbdt"]),
            "num_threads": ("none", [1]),
            "num_boost_round": ("int", {"low": 20, "high": 1000, "log": False}),
        }
        self.best_val_loss = float('inf')
        self.best_params = None
        self.best_iter = None
        self.best_model = None

    def sample(self):
        params = {}
        for k, (ptype, spec) in self.param_space.items():
            if ptype == "float":
                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    val = float(np.exp(np.random.uniform(np.log(low), np.log(high))))
                else:
                    val = float(np.random.uniform(low, high))
                params[k] = val
            elif ptype == "int":
                low, high = spec["low"], spec["high"]
                val = int(np.random.randint(low, high + 1))
                params[k] = val
            elif ptype == "categorical":
                val = np.random.choice(spec)
                params[k] = val
            elif ptype == "none":
                params[k] = spec[0]
        return params
    
    def update(self, loss, params, model=None, iter_num=None):
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_params = params.copy()
            self.best_iter = iter_num
            if model is not None:
                self.best_model = model


class LocalModel:
    def __init__(self, model: str, data_info, qlevels =[0.5, 0.8, 0.9, 0.95, 0.99]) -> None:
        assert model in ["ISQ", "iETS", "tweedieGP", "MW"]
        self.model = model
        self.h = data_info['h']
        self.qlevels = qlevels
        if self.model == "tweedieGP":
            x = torch.arange(data_info['len']).to(torch.float32)
            x =  x/{"D":365, "W":52, "M":12}[data_info['freq']]
            self.train_x = x[:-data_info['h']]
            self.test_x = x[-data_info['h']:]
            self.tweediegp = intermittentGP(
                likelihood = "tweedie",
                scaling = "median-demand",
                num_inducing_points= None if data_info['len'] < 200 else 200,
                n_samples=10000
            )
        if self.model == "iETS":
            logging.getLogger("rpy2").setLevel(logging.ERROR)
            logging.getLogger("rpy2.situation").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface_lib").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface_lib.embedded").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface").setLevel(logging.ERROR)
            from rpy2 import robjects
            from rpy2.robjects.packages import isinstalled
            self.robjects = robjects
            if not isinstalled("smooth"):
                logging.info("R package 'smooth' is missing; installing it now.")
                self.robjects.r("install.packages('smooth', repos='https://cloud.r-project.org')")
            if not isinstalled("forecast"):
                logging.info("R package 'forecast' is missing; installing it now.")
                self.robjects.r("if (!requireNamespace('forecast', quietly=TRUE)) install.packages('forecast', repos='https://cloud.r-project.org')")
            try:
                import rpy2.rinterface_lib.callbacks as rpy2_callbacks
                rpy2_callbacks.logger.setLevel("ERROR")  # silence embedded R chatter
                rpy2_callbacks.consolewrite_print = lambda _: None
            except Exception:
                pass
            self.iets = self.robjects.r(f"""
                function(train_y_R) {{
                    set.seed(0)
                    suppressWarnings(model <- smooth::adam(train_y_R, model="MNN", occurrence="auto"))
                    suppressWarnings(pred <- forecast::forecast(model, h={self.h}, level=c({", ".join([str(q) for q in self.qlevels])}), 
                                                                interval='simulated', nsim=10000, scenarios=TRUE, side='upper'))
                    list(as.numeric(pred$mean), pred$upper)
                }}
            """)
        if self.model == "MW":
            logging.getLogger("rpy2.situation").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface_lib").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface_lib.embedded").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface").setLevel(logging.ERROR)
            from rpy2 import robjects
            seasonality = {"D":365, "W":52, "M":12}[data_info['freq']]
            self.robjects = robjects
            # self.robjects.r("if (!requireNamespace('smooth', quietly=TRUE)) install.packages('smooth', repos='https://cloud.r-project.org')")
            # self.robjects.r("if (!requireNamespace('forecast', quietly=TRUE)) install.packages('forecast', repos='https://cloud.r-project.org')")
            try:
                import rpy2.rinterface_lib.callbacks as rpy2_callbacks
                rpy2_callbacks.logger.setLevel("ERROR")
                rpy2_callbacks.consolewrite_print = lambda _: None
            except Exception:
                pass
            self.mw = self.robjects.r(f"""
                function(train_y_R) {{
                                    Bernoulli <- function(y, steps) {{
                                        alpha <- -1
                                        co <- mean(y[y > 0])
                                        bern <- y
                                        bern[bern > 0] <- 1
                                        su <- bern[-1] + bern[-length(y)]
                                        di <- bern[-1] - bern[-length(y)]
                                        n00 <- length(su[su == 0])
                                        n11 <- length(su[su == 2])
                                        n01 <- length(di[di == -1])
                                        n10 <- length(di[di == 1])
                                        p00 <- n00 / (n00 + n10)
                                        xi  <- n10 / (n10 + n00)
                                        p   <- mean(bern)
                                        lambda <- n11 / (n11 + n01)
                                        if (n00 == 0 & n10 == 0) lambda <- 1
                                        if (n11 == 0 & n01 == 0) lambda <- 0
                                        if (lambda <= 0) lambda <- .0001
                                        delta <- p00 + lambda - 1
                                        if (lambda == 1) {{
                                            p <- 1
                                            xi <- 0
                                            delta <- 1
                                        }}

                                        MC <- c()
                                        MC[1] <- xi + delta * tail(bern, 1)
                                        for (s in 2:steps) {{
                                            MC[s] <- xi + delta * MC[s - 1]
                                        }}

                                        m <- v <- c()
                                        m[1] <- 0
                                        k <- (lambda^2 - 1 + sqrt(1 - lambda^2)) / lambda

                                        for (t in 1:length(y)) {{
                                            v[t] <- y[t] - bern[t] * co - m[t]
                                            m[t + 1] <- lambda * m[t] + k * v[t]
                                        }}

                                        fo <- c()
                                        for (s in 1:steps) {{
                                            fo[s] <- co * MC[s] + lambda^(s - 1) * m[length(m)]
                                        }}

                                        list(bern, p, lambda, MC, delta, xi, fo, v, k)
                                    }}

                                    MW <- function(y, steps, qlevels) {{

                                        mysum <- function(x, steps) {{
                                            d <- 0
                                            for (f in 0:(steps - 2)) d <- d + x^(f * 2)
                                            d
                                        }}

                                        s <- {seasonality}
                                        if ((length(y[y == 0]) / length(y)) < .95 && length(y) >= s) {{
                                            h <- steps
                                            cma <- matrix(NA, length(y), 1)

                                            for (g in 1:(length(y) - s + 1)) {{
                                                cma[g + ((s + 1) / 2) - 1] <- mean(y[g:(g + s - 1)])
                                            }}

                                            residuals <- y / cma

                                            sfactors <- c()
                                            for (seas in 1:s) {{
                                                sfactors[seas] <- mean(na.omit(residuals[seq(seas, length(y) - s + seas, by = s)]))
                                            }}

                                            sfactout <- rep(sfactors, length(y) + h)[(length(y) + 1):(length(y) + h)]
                                            y <- y / rep(sfactors, ceiling(length(y) / s))[1:length(y)]
                                            y[is.na(y)] <- 0
                                            y[y == Inf] <- 0

                                        }} else {{
                                            sfactout <- rep(1, steps)
                                        }}

                                        h <- c()
                                        le <- max(1, floor({self.h} / 2))
                                        max_iter <- ceiling(length(y) / le)

                                        for (i in 0:max_iter) {{
                                            Y <- tail(y, (length(y) - i * le))
                                            if (length(Y) < le) break
                                            ins <- head(Y, length(Y) - steps)

                                            if ((length(ins[ins == 0]) / length(ins)) < .99 & length(ins) > 100) {{
                                                h[i + 1] <- mean((tail(Y, steps) - Bernoulli(ins, steps)[[7]])^2)
                                            }}
                                        }}

                                        if (length(h) != 0) {{
                                            y <- tail(y, (length(y) - which.min(h[!is.na(h)]) * le) + steps)
                                        }}

                                        co <- mean(y[y > 0])
                                        ma <- Bernoulli(y, steps)

                                        bern   <- ma[[1]]
                                        p      <- ma[[2]]
                                        lambda <- ma[[3]]
                                        MC     <- ma[[4]]
                                        delta  <- ma[[5]]
                                        fo     <- ma[[7]]
                                        v      <- ma[[8]]
                                        k      <- ma[[9]]

                                        fo <- fo * sfactout

                                        if (p < 1) {{
                                            vari <- ((-1 + lambda) * (1 + lambda - 2 * p) * p) / (-1 + p)
                                        }} else {{
                                            vari <- 0
                                        }}

                                        Interv <- c()
                                        Interv[1] <- vari * co^2 + var(v)
                                        for (j in 2:steps) {{
                                            Interv[j] <- vari * mysum(delta, j) * co^2 +
                                                                     (var(v) * (1 + k^2 * (mysum(lambda, j))))
                                        }}

                                        qlevels[qlevels <= 0] <- 1e-4
                                        qlevels[qlevels >= 1] <- 1 - 1e-4
                                        z <- qnorm(qlevels)
                                        quantiles <- sapply(z, function(zz) fo + zz * sqrt(Interv))

                                        list(
                                            mean = fo,
                                            quantiles = quantiles
                                        )
                                    }}

                                    set.seed(0)
                                    pred <- MW(train_y_R, {self.h}, c({", ".join([str(q) for q in self.qlevels])}))
                                    list(as.numeric(pred$mean), pred$quantiles)
                                    }}
                                """)
        

    def forecast(self, train_y):
        # mena forecast are h-dimensional, quantiles_forecast are h-dimensional x len(qlevels)
        mean_forecast, quantile_forecasts = None, None
        if self.model == 'ISQ':
            mean_forecast = np.repeat(np.mean(train_y), repeats=self.h)
            quantile_forecasts = np.tile(np.quantile(train_y, self.qlevels), (self.h,1))
        if self.model == 'iETS':
            train_y_R = self.robjects.FloatVector(train_y)
            iets_fore = self.iets(train_y_R)
            mean_forecast = np.array(iets_fore.rx2(1))
            quantile_forecasts = np.array(iets_fore.rx2(2))
        if self.model == 'tweedieGP':
            torch.manual_seed(0)
            train_y = torch.tensor(train_y, dtype=torch.float32)
            self.tweediegp.build(self.train_x, train_y)
            self.tweediegp.fit(self.train_x, train_y)
            mean_forecast, samples = self.tweediegp.predict(self.test_x)
            mean_forecast = mean_forecast.detach().numpy()
            quantile_forecasts = np.quantile(samples.detach().numpy(), self.qlevels, axis=0).T
        if self.model == 'MW':
            train_y_R = self.robjects.FloatVector(train_y)
            mw_fore = self.mw(train_y_R)
            mean_forecast = np.array(mw_fore.rx2(1))
            quantile_forecasts = np.array(mw_fore.rx2(2))
        return(mean_forecast, quantile_forecasts)

