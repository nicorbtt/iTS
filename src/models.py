import os
import logging
import numpy as np
import torch

torch.set_num_threads(1)

# NOTE: gluonts/transformers/lightgbmlss/tweediegp are NOT imported at module level.
# They're only needed by ModelConfigBuilder/LocalModel (global/local models, which rely
# on this project's forked distribution heads); FoundationModel needs none of them. Each
# is imported lazily where used instead, so `from models import FoundationModel` works in
# any environment, including ones without these forks installed (e.g. environment_foundation.yml).

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
        from gluonts.time_feature import time_features_from_frequency_str, get_lags_for_frequency
        from gluonts.torch.distributions import (
            PoissonOutput,
            NegativeBinomialOutput,
            TweedieOutput,
            FixedDispersionTweedieOutput,
            HurdleShiftedPoissonOutput,
            HurdleShiftedNegativeBinomialOutput,
        )
        from transformers import TimeSeriesTransformerConfig, InformerConfig, AutoformerConfig

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
        from gluonts.torch.model.deepar.module import DeepARModel
        from gluonts.torch.model.simple_feedforward.module import SimpleFeedForwardModel
        from gluonts.torch.model.d_linear.module import DLinearModel
        from gluonts.torch.model.patch_tst.module import PatchTSTModel
        from gluonts.torch.model.tide.module import TiDEModel
        from transformers import TimeSeriesTransformerForPrediction, InformerForPrediction, AutoformerForPrediction
        from lightgbmlss.model import LightGBMLSS
        from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
        from lightgbmlss.distributions.Tweedie import TweedieDistribution
        from lightgbmlss.distributions.HurdleShiftedNegativeBinomial import HurdleShiftedNegativeBinomial

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
                        'tweedie' : TweedieDistribution(response_fn_mu='softplus', response_fn_phi='softplus', response_fn_rho='sigmoid_rho'),
                        'hsnb' : HurdleShiftedNegativeBinomial(response_fn_total_count='softplus', response_fn_logits='identity', response_fn_p_zero='sigmoid')
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
    from gluonts.torch.model.deepar.module import DeepARModel
    from gluonts.torch.model.simple_feedforward.module import SimpleFeedForwardModel
    from gluonts.torch.model.d_linear.module import DLinearModel
    from gluonts.torch.model.patch_tst.module import PatchTSTModel
    from gluonts.torch.model.tide.module import TiDEModel
    from transformers import TimeSeriesTransformerForPrediction, InformerForPrediction, AutoformerForPrediction
    from lightgbmlss.model import LightGBMLSS

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
    from gluonts.torch.model.deepar.module import DeepARModel
    from gluonts.torch.model.simple_feedforward.module import SimpleFeedForwardModel
    from gluonts.torch.model.d_linear.module import DLinearModel
    from gluonts.torch.model.patch_tst.module import PatchTSTModel
    from gluonts.torch.model.tide.module import TiDEModel
    from transformers import TimeSeriesTransformerForPrediction, InformerForPrediction, AutoformerForPrediction
    from lightgbmlss.model import LightGBMLSS

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
            "learning_rate": ("float", {"low": 5e-3, "high": 0.1, "log": True}),
            "max_depth": ("int", {"low": 4, "high": 10, "log": False}),
            "num_leaves": ("int", {"low": 2, "high": 200, "log": True}),
            "min_data_in_leaf": ("int", {"low": 1, "high": 64, "log": True}),
            "min_gain_to_split": ("float", {"low": 1e-8, "high": 40.0, "log": False}),
            "subsample": ("float", {"low": 0.7, "high": 1.0, "log": False}),
            "feature_fraction": ("float", {"low": 0.4, "high": 1.0, "log": False}),
            "num_boost_round": ("int", {"low": 50, "high": 1000, "log": False}),
            "bagging_freq": ("int", {"low": 1, "high": 1, "log": False}),
            "extra_trees": ("categorical", [False, True]),
            "boosting": ("categorical", ["gbdt"]),
            "device": ("categorical", ["cpu"]),
            "num_threads": ("none", [1]),
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
        assert model in ["ISQ", "iETS", "tweedieGP", "MW", "gasNB"]
        self.model = model
        self.h = data_info['h']
        self.qlevels = qlevels
        if self.model == "tweedieGP":
            from tweediegp.intermittent_gp import intermittentGP
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
                                            if (!any(is.na(sfactors))){{
                                                sfactout <- rep(sfactors, length(y) + h)[(length(y) + 1):(length(y) + h)]
                                                y <- y / rep(sfactors, ceiling(length(y) / s))[1:length(y)]
                                                y[is.na(y)] <- 0
                                                y[y == Inf] <- 0
                                            }} else {{
                                                sfactout <- rep(1, steps)
                                            }}
                                        }} else {{
                                            sfactout <- rep(1, steps)
                                        }}
                                        h <- c()
                                        le <- max(1, floor(steps / 2))
                                        max_iter <- ceiling(length(y) / le)
                                        for (i in 0:max_iter) {{
                                            Y <- tail(y, (length(y) - i * le))
                                            if (length(Y) < le) break
                                            ins <- head(Y, length(Y) - steps)
                                            if ((length(ins[ins == 0]) / length(ins)) < .99 & length(ins) > (4*steps)) {{
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
                                            Interv[j] <- vari * mysum(delta, j) * co^2 + (var(v) * (1 + k^2 * (mysum(lambda, j))))
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
        if self.model == "gasNB":
            logging.getLogger("rpy2").setLevel(logging.ERROR)
            logging.getLogger("rpy2.situation").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface_lib").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface_lib.embedded").setLevel(logging.ERROR)
            logging.getLogger("rpy2.rinterface").setLevel(logging.ERROR)
            from rpy2 import robjects
            from rpy2.robjects.packages import isinstalled
            period_len = {"D": 7, "W": 52, "M": 12}[data_info['freq']]
            self.robjects = robjects
            if not isinstalled("nloptr"):
                logging.info("R package 'nloptr' is missing; installing it now.")
                self.robjects.r("install.packages('nloptr', repos='https://cloud.r-project.org')")
            try:
                import rpy2.rinterface_lib.callbacks as rpy2_callbacks
                rpy2_callbacks.logger.setLevel("ERROR")
                rpy2_callbacks.consolewrite_print = lambda _: None
            except Exception:
                pass
            self.gasnb = self.robjects.r(f"""
                function(train_y_R) {{
                    epsilon <- 1e-4
                    period_len <- {period_len}
                    h <- {self.h}

                    gas_filter <- function(y, period_len, psi0, phi, rho, xi0, k, alpha) {{
                        n <- length(y)
                        psi <- psi0
                        xi <- rep(xi0, period_len)
                        f <- numeric(n)

                        for (t in 1:n) {{
                            if (t == 1) {{
                                score <- 0
                            }} else {{
                                score <- (y[t - 1] - f[t - 1]) / (1 + f[t - 1] / alpha)
                            }}

                            psi <- phi * psi + rho * score

                            if (period_len > 1) {{
                                season <- ((t - 1) %% period_len) + 1
                                past_season <- ifelse(season == 1, period_len, season - 1)
                                xi[past_season] <- xi[past_season] + k * score
                                xi[-past_season] <- xi[-past_season] - (k / (period_len - 1)) * score
                                gamma <- xi[season]
                                f[t] <- exp(psi + gamma)
                            }} else {{
                                f[t] <- exp(psi)
                            }}
                        }}

                        list(f = f, last_psi = psi, last_xi = xi)
                    }}

                    nll <- function(y, period_len, psi0, phi, rho, xi0, k, alpha) {{
                        res <- gas_filter(y, period_len, psi0, phi, rho, xi0, k, alpha)
                        f <- res$f
                        l <- -sum(dnbinom(y, size = alpha, mu = f, log = TRUE))
                        if (any(!is.finite(f)) || !is.finite(l)) {{
                            return(1 / epsilon)
                        }} else {{
                            return(l)
                        }}
                    }}

                    mean_y <- mean(train_y_R)
                    max_y <- max(train_y_R)
                    lb <- c(log(epsilon), -1 + epsilon, epsilon, log(epsilon), epsilon, epsilon)
                    ub <- c(log(max(max_y, epsilon)), 1 - epsilon, 1 - epsilon, log(max(max_y, epsilon)), 1 - epsilon, Inf)
                    x0 <- c(log(max(mean_y, epsilon)) * 2 / 3, 0, 0.1, log(max(mean_y, epsilon)) / 3, 0.1, max(mean_y, epsilon))

                    eval_f <- function(x) {{
                        value <- tryCatch(
                            nll(train_y_R, period_len, x[1], x[2], x[3], x[4], x[5], x[6]),
                            error = function(e) 1 / epsilon
                        )
                        if (!is.finite(value)) {{
                            value <- 1 / epsilon
                        }}
                        value
                    }}

                    sol <- nloptr::nloptr(
                        x0 = x0,
                        eval_f = eval_f,
                        lb = lb,
                        ub = ub,
                        opts = list(
                            algorithm = "NLOPT_LN_COBYLA",
                            ftol_rel = 1e-4,
                            maxeval = 500,
                            print_level = 0
                        )
                    )

                    if (!is.finite(sol$objective) || sol$objective == 1 / epsilon) {{
                        stop("Optimization failed to find a finite solution.")
                    }}

                    psi <- sol$solution[1]
                    phi <- sol$solution[2]
                    rho <- sol$solution[3]
                    xi0 <- sol$solution[4]
                    k <- sol$solution[5]
                    alpha <- sol$solution[6]

                    filter <- gas_filter(train_y_R, period_len, psi, phi, rho, xi0, k, alpha)

                    n_samples <- 10000
                    f_state <- rep(filter$f[length(train_y_R)], n_samples)
                    psi_state <- rep(filter$last_psi, n_samples)
                    xi_state <- matrix(rep(filter$last_xi, n_samples), nrow = period_len)
                    y_state <- rep(train_y_R[length(train_y_R)], n_samples)

                    fc_samples <- matrix(NA, nrow = n_samples, ncol = h)
                    for (i in 1:h) {{
                        score <- (y_state - f_state) / (1 + f_state / alpha)
                        psi_state <- phi * psi_state + rho * score

                        if (period_len > 1) {{
                            season <- ((length(train_y_R) + i - 1) %% period_len) + 1
                            past_season <- ifelse(season == 1, period_len, season - 1)
                            xi_state[past_season, ] <- xi_state[past_season, ] + k * score
                            xi_state[-past_season, ] <- xi_state[-past_season, ] - k / (period_len - 1) * score
                            gamma <- xi_state[season, ]
                            f_state <- exp(psi_state + gamma)
                        }} else {{
                            f_state <- exp(psi_state)
                        }}

                        y_state <- rnbinom(n_samples, size = alpha, mu = f_state)
                        fc_samples[, i] <- y_state
                    }}

                    mean_fc <- colMeans(fc_samples)
                    quantile_fc <- apply(fc_samples, 2, quantile, probs = c({", ".join([str(q) for q in self.qlevels])}))
                    list(mean_fc, t(quantile_fc))
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
            train_y = torch.tensor(train_y, dtype=torch.float32)
            max_attempts = 5
            last_error = None

            for attempt in range(max_attempts):
                torch.manual_seed(attempt)
                self.tweediegp.build(self.train_x, train_y)
                try:
                    self.tweediegp.fit(self.train_x, train_y)
                    mean_forecast, samples = self.tweediegp.predict(self.test_x)
                    last_error = None
                    break
                except Exception as err:
                    last_error = err
            if last_error is not None:
                raise last_error
            mean_forecast = mean_forecast.detach().numpy()
            quantile_forecasts = np.quantile(samples.detach().numpy(), self.qlevels, axis=0).T
        if self.model == 'MW':
            train_y_R = self.robjects.FloatVector(train_y)
            mw_fore = self.mw(train_y_R)
            mean_forecast = np.array(mw_fore.rx2(1))
            quantile_forecasts = np.array(mw_fore.rx2(2))
        if self.model == 'gasNB':
            train_y_R = self.robjects.FloatVector(train_y)
            gasnb_fore = self.gasnb(train_y_R)
            mean_forecast = np.array(gasnb_fore.rx2(1))
            quantile_forecasts = np.array(gasnb_fore.rx2(2))
        return(mean_forecast, quantile_forecasts)


class FoundationModel:
    """Common interface over pretrained (zero-shot) time series foundation models.

    Unlike ModelConfigBuilder/LocalModel above, this class never needs the project's
    forked gluonts/transformers/lightgbmlss/tweediegp packages -- none of the foundation
    models use their custom distribution heads. Combined with the lazy imports used
    elsewhere in this file, `from models import FoundationModel` works in any
    environment, including ones without these forks installed (see environment_foundation.yml).
    """

    _SECONDS_PER_STEP = {"D": 86400, "W": 604800, "M": 2629746}

    def __init__(self, model: str, data_info, num_samples=100,
                 qlevels=[0.5, 0.8, 0.9, 0.95, 0.99], device="cpu") -> None:
        assert model in ["chronos2", "toto", "toto2", "timesfm", "tirex"]
        self.model = model
        self.h = data_info['h']
        self.freq = data_info['freq']
        self.qlevels = qlevels
        self.num_samples = num_samples
        self.device = device

        if self.model == "chronos2":
            from chronos import Chronos2Pipeline
            self.chronos2 = Chronos2Pipeline.from_pretrained(
                "autogluon/chronos-2-small", device_map=device, torch_dtype=torch.float32
            )

        if self.model == "timesfm":
            # device is not configurable here: TimesFM auto-selects cuda if available, else cpu
            import timesfm
            self.timesfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            self.timesfm.compile(
                timesfm.ForecastConfig(
                    max_context=data_info['len'] - data_info['h'],
                    max_horizon=self.h,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )

        if self.model == "tirex":
            from tirex import load_model
            self.tirex = load_model("NX-AI/TiRex", device=device)

        if self.model == "toto2":
            from toto2 import Toto2Model
            self.toto2 = Toto2Model.from_pretrained("Datadog/Toto-2.0-4m")
            self.toto2 = self.toto2.to(device).eval()
            self.toto2_patch_size = self.toto2.config.patch_size

        if self.model == "toto":
            from toto.model.toto import Toto
            from toto.inference.forecaster import TotoForecaster
            from toto.data.util.dataset import MaskedTimeseries
            self.MaskedTimeseries = MaskedTimeseries
            self.toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
            self.toto.to(device)
            self.toto_forecaster = TotoForecaster(self.toto.model)
            self.seconds_per_step = self._SECONDS_PER_STEP[self.freq]

    def forecast(self, train_y_batch):
        # train_y_batch: list of B 1-D array-likes (equal length within a dataset)
        # returns mean_forecast [B,h] and quantile_forecasts [B,h,len(qlevels)]
        train_y_batch = [np.asarray(y, dtype=np.float32) for y in train_y_batch]
        mean_forecast, quantile_forecasts = None, None

        if self.model == "chronos2":
            context = [torch.tensor(y) for y in train_y_batch]
            quantiles, mean = self.chronos2.predict_quantiles(
                context, prediction_length=self.h, quantile_levels=self.qlevels
            )
            mean_forecast = np.stack([m.numpy().squeeze(0) for m in mean], axis=0)             # [B,h]
            quantile_forecasts = np.stack([q.numpy().squeeze(0) for q in quantiles], axis=0)   # [B,h,Q]

        if self.model == "timesfm":
            assert self.qlevels == [0.5, 0.8, 0.9, 0.95, 0.99]
            point_forecast, quantile_forecast = self.timesfm.forecast(horizon=self.h, inputs=train_y_batch)
            mean_forecast = point_forecast                              # [B,h]
            deciles = quantile_forecast[:, :, 1:10]                     # [B,h,9], columns = q0.1..q0.9 in order
            q05, q08, q09 = deciles[:, :, 4], deciles[:, :, 7], deciles[:, :, 8]
            # TimesFM only outputs deciles up to q0.9: q0.95/q0.99 are NOT genuine model
            # output, but a linear extrapolation of the q0.8-q0.9 gap, per user's request
            gap = q09 - q08
            q095 = q09 + gap * 0.5
            q099 = q09 + gap * 0.9
            quantile_forecasts = np.stack([q05, q08, q09, q095, q099], axis=-1)  # [B,h,5]

        if self.model == "tirex":
            assert self.qlevels == [0.5, 0.8, 0.9, 0.95, 0.99]
            context = [torch.tensor(y) for y in train_y_batch]
            deciles, mean = self.tirex.forecast(context=context, prediction_length=self.h)
            mean_forecast = mean.numpy()                              # [B,h]
            deciles = deciles.numpy()                                 # [B,h,9], columns = q0.1..q0.9 in order
            q05, q08, q09 = deciles[:, :, 4], deciles[:, :, 7], deciles[:, :, 8]
            # TiRex only outputs deciles up to q0.9: q0.95/q0.99 are NOT genuine model
            # output, but a linear extrapolation of the q0.8-q0.9 gap, same as timesfm above
            gap = q09 - q08
            q095 = q09 + gap * 0.5
            q099 = q09 + gap * 0.9
            quantile_forecasts = np.stack([q05, q08, q09, q095, q099], axis=-1)  # [B,h,5]

        if self.model == "toto2":
            assert self.qlevels == [0.5, 0.8, 0.9, 0.95, 0.99]
            # Toto2 needs the context length to already be a multiple of patch_size before
            # the call (it doesn't pad internally) -- left-pad and mask the padding out.
            # Batching here was verified safe (unlike toto 1.0): forecasting a series alone
            # vs. alongside unrelated companions gave numerically identical results.
            T = len(train_y_batch[0])
            B = len(train_y_batch)
            pad = (-T) % self.toto2_patch_size
            padded = np.stack([np.concatenate([np.zeros(pad, dtype=np.float32), y]) for y in train_y_batch])
            target = torch.tensor(padded, dtype=torch.float32).unsqueeze(1)  # [B,1,T+pad]
            target_mask = torch.zeros_like(target, dtype=torch.bool)
            target_mask[:, :, pad:] = True
            series_ids = torch.zeros(B, 1, dtype=torch.long)
            quantiles = self.toto2.forecast(
                {"target": target, "target_mask": target_mask, "series_ids": series_ids},
                horizon=self.h,
                decode_block_size=None,
                has_missing_values=(pad > 0),
            )  # [9, B, 1, h]
            quantiles = quantiles[:, :, 0, :].detach().cpu().numpy()  # [9, B, h]
            q05, q08, q09 = quantiles[4], quantiles[7], quantiles[8]
            # Toto2 has no separate point-forecast output: use the median as mean_forecast,
            # same convention as chronos2 (mean == q0.5 there too).
            mean_forecast = q05
            # Toto2 only outputs deciles up to q0.9: q0.95/q0.99 are NOT genuine model
            # output, but a linear extrapolation of the q0.8-q0.9 gap, same as timesfm/tirex above
            gap = q09 - q08
            q095 = q09 + gap * 0.5
            q099 = q09 + gap * 0.9
            quantile_forecasts = np.stack([q05, q08, q09, q095, q099], axis=-1)  # [B,h,5]

        if self.model == "toto":
            # NOTE: batching multiple series in one MaskedTimeseries call was verified (empirically,
            # holding the random seed fixed and averaging over many samples/seeds) to leak information
            # across unrelated series in the batch -- forecasts for the same series changed depending on
            # which other series shared the call. So, unlike chronos2/timesfm, each series is forecast
            # in its own batch-of-1 call here, even though the outer interface still takes/returns a batch.
            T = len(train_y_batch[0])
            mean_list, quantile_list = [], []
            for y in train_y_batch:
                series = torch.tensor(y).view(1, 1, T).to(self.device)  # [1,1,T]
                padding_mask = torch.ones_like(series, dtype=torch.bool)
                id_mask = torch.zeros((1, 1, T), dtype=torch.int32, device=self.device)
                timestamp_seconds = (torch.arange(T, dtype=torch.int64) * int(self.seconds_per_step)).view(1, 1, T).to(self.device)
                time_interval_seconds = torch.full((1, 1), int(self.seconds_per_step), dtype=torch.int64).to(self.device)
                inputs = self.MaskedTimeseries(
                    series=series,
                    padding_mask=padding_mask,
                    id_mask=id_mask,
                    timestamp_seconds=timestamp_seconds,
                    time_interval_seconds=time_interval_seconds,
                )
                forecast = self.toto_forecaster.forecast(
                    inputs, prediction_length=self.h, num_samples=self.num_samples, samples_per_batch=self.num_samples
                )
                samples = forecast.samples.squeeze(1).squeeze(0).detach().cpu().numpy()  # [h, num_samples]
                mean_list.append(np.maximum(samples.mean(axis=-1), 0))  # clip: demand is non-negative, occasional extreme samples can pull the sample mean below 0
                quantile_list.append(np.quantile(samples, self.qlevels, axis=-1).T)
            mean_forecast = np.stack(mean_list, axis=0)            # [B,h]
            quantile_forecasts = np.stack(quantile_list, axis=0)   # [B,h,Q]

        return (mean_forecast, quantile_forecasts)

