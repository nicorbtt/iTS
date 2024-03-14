import numpy as np
import torch
from gluonts.torch.model.deepar.module import DeepARModel
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
    FixedDispersionTweedieOutput
)
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction



### Configuration dictionary
class ModelConfigBuilder:
    
    def __init__(self, model, distribution_head, scaling):
        assert model in ["deepAR", "transformer"]
        assert distribution_head in ["poisson","negative_binomial", "tweedie", "tweedie-fix"]
        assert scaling in ["mase", "mean", "mean-demand", None]
        self.model = model
        self.distribution_head = distribution_head
        self.scaling = scaling

    def build(self, data_info, **kwargs) -> None:
        def __check__(key, default_value):
            return kwargs[key] if key in kwargs else default_value
        
        lags_sequence = get_lags_for_frequency(data_info['freq'] if data_info['freq'] != "M" else "ME") if not 'lag_sequence' in kwargs else []
        self.time_features = time_features_from_frequency_str(data_info['freq'] if data_info['freq'] != "M" else "ME") if not 'time_features' in kwargs else []
        
        if self.model == "deepAR":
            self.params = {
                'freq' : data_info['freq'],
                'context_length' : __check__('context_length', data_info['h']*2),
                'prediction_length' : __check__('prediction_length', data_info['h']),
                'num_feat_dynamic_real' : 0,
                'num_feat_static_real' : 0,
                'num_feat_static_cat' : 1,
                'cardinality' : [data_info['N']],
                'embedding_dimension' : [__check__('embedding_dimension', 3)],
                'num_layers' : __check__('num_layers', 2),
                'hidden_size' : __check__('hidden_size', 40),
                'dropout_rate' : __check__('dropout_rate', 0.1),
                'distr_output' : {
                        'poisson' : PoissonOutput(),
                        'negative_binomial' : NegativeBinomialOutput(),
                        'tweedie' : TweedieOutput(),
                        'tweedie-fix' : FixedDispersionTweedieOutput()
                    }[self.distribution_head],
                'lags_seq' : lags_sequence,
                'scaling' : {
                        'MASE' : 'MASE',
                        'mean' : 'mean',
                        'mean-demand' : 'mean demand'
                    }[self.scaling] if self.scaling else None,
                'default_scale' : None,
                'num_parallel_samples' : __check__('num_parallel_samples', 100)
            }

        if self.model == "transformer":
            self.params = TimeSeriesTransformerConfig(
                prediction_length = __check__('prediction_length', data_info['h']),
                context_length = __check__('context_length', data_info['h']*2),
                distribution_output = {
                        'poisson' : 'poisson',
                        'negative_binomial' : 'negative_binomial',
                        'tweedie' : 'tweedie',
                        'tweedie-fix' : 'fixed dispersion tweedie'
                    }[self.distribution_head],
                loss = "nll",
                input_size = 1,
                scaling = {
                        'MASE' : 'MASE',
                        'mean' : 'mean',
                        'mean-demand' : 'mean demand'
                    }[self.scaling] if self.scaling else None,
                lags_sequence = lags_sequence,
                num_time_features = len(self.time_features) + 1,  # +1 is Age
                num_dynamic_real_features = 0,
                num_static_categorical_features = 1,
                num_static_real_features = 0,
                cardinality = [data_info['N']],
                embedding_dimension = [__check__('embedding_dimension', 3)],
                
                # architecture params
                d_model = __check__('d_model', 32),                                    # Dimensionality of the transformer layers                   
                encoder_layers = __check__('encoder_layers', 4),                       # Number of encoder layers              
                decoder_layers = __check__('decoder_layers', 4),                       # Number of decoder layers                                          
                encoder_attention_heads = __check__('encoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer encoder         
                decoder_attention_heads = __check__('decoder_attention_heads', 2),     # Number of attention heads for each attention layer in the Transformer decoder   
                encoder_ffn_dim = __check__('encoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in encoder                
                decoder_ffn_dim = __check__('decoder_ffn_dim', 32),                    # Dimension of the “intermediate” (often named feed-forward) layer in decoder 
                activation_function = __check__('activation_function', "gelu"),        # The non-linear activation function (function or string) in the encoder and decoder
                dropout = __check__('dropout', 0.1),                                   # The dropout probability for all fully connected layers in the encoder, and decoder
                encoder_layerdrop = __check__('encoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each encoder layer  
                decoder_layerdrop = __check__('decoder_layerdrop', 0.1),               # The dropout probability for the attention and fully connected layers for each decoder layer
                attention_dropout = __check__('attention_dropout', 0.1),               # The dropout probability for the attention probabilities
                activation_dropout = __check__('activation_dropout', 0.1),             # The dropout probability used between the two layers of the feed-forward networks
                num_parallel_samples = __check__('num_parallel_samples', 100),         # The number of samples to generate in parallel for each time step of inference
                init_std = __check__('init_std', 0.02),                                # The standard deviation of the truncated normal weight initialization distribution
                use_cache = __check__('use_cache', True),                              # Whether to use the past key/values attentions (if applicable to the model) to speed up decoding
            )

    ### Create Model
    def get_model(self):
        if self.model == "deepAR" : 
            tmp = self.params['num_feat_dynamic_real'] + len(self.time_features) + 1
            return(DeepARModel(**({**self.params, 'num_feat_dynamic_real': tmp, 'num_feat_static_real':1})))
        if self.model == "transformer" : 
            return(TimeSeriesTransformerForPrediction(self.params))


### Forward step
def forward(model, batch, device, config):
    loss = None
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
    return(loss)

### Generate forecasts
def predict(model, batch, device, config):
    predictions = None
    if isinstance(model, TimeSeriesTransformerForPrediction):
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
        ).detach().numpy()
    # TODO rounding
    return(predictions)

class EarlyStop():
    def __init__(self, logger, patience=20, min_delta = 0.001) -> None:
        self.best_val_loss = np.inf
        self.best_model = None
        self.current_patience = 0
        self.logger = logger
        self.patience = patience
        self.min_delta = min_delta

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
                return True
        return False

