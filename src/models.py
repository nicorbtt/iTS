import numpy as np
import torch
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
from rpy2 import robjects

from tweediegp.intermittent_gp import intermittentGP

### Configuration dictionary
class ModelConfigBuilder:
    
    def __init__(self, model, distribution_head, scaling):
        assert model in ["deepAR", "transformer", "informer", "autoformer", "patchTST", "tide", "dlinear", "feedforward"]
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
                # Keep base config minimal; runtime dimensions are adapted in get_model.
                'num_feat_dynamic_real' : 0,
                'num_feat_dynamic_proj' : _check('num_feat_dynamic_proj', 2),
                'num_feat_static_real' : 1,
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
    return(loss)

### Generate forecasts
def predict(model, batch, device, config):
    predictions = None
    def _ensure_channel_dim(tensor):
        return tensor.unsqueeze(-1) if tensor.dim() == 2 else tensor
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


class LocalModel:
    def __init__(self, model: str, data_info, qlevels =[0.5, 0.8, 0.9, 0.95, 0.99]) -> None:
        assert model in ["ISQ", "iETS", "tweedieGP"]
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
            from rpy2 import robjects
            try:
                import rpy2.rinterface_lib.callbacks as rpy2_callbacks
                rpy2_callbacks.logger.setLevel("ERROR")  # silence embedded R chatter
                rpy2_callbacks.consolewrite_print = lambda _: None
            except Exception:
                pass
            self.robjects = robjects
            self.iets = self.robjects.r(f"""
                function(train_y_R) {{
                    set.seed(0)
                    suppressWarnings(model <- smooth::adam(train_y_R, model="MNN", occurrence="auto"))
                    suppressWarnings(pred <- forecast::forecast(model, h={self.h}, level=c({", ".join([str(q) for q in self.qlevels])}), 
                                                                interval='simulated', nsim=10000, scenarios=TRUE, side='upper'))
                    list(as.numeric(pred$mean), pred$upper)
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
        return(mean_forecast, quantile_forecasts)

