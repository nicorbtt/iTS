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
class ModelConfig(TimeSeriesTransformerConfig):
    def __init__(self, datasets, data_info, model, **kwargs) -> None:
        assert model in ["deepAR", "transformer"]

        lags_sequence = get_lags_for_frequency(data_info['freq'] if data_info['freq'] != "M" else "ME") if not 'lag_sequence' in kwargs else None
        time_features = time_features_from_frequency_str(data_info['freq'] if data_info['freq'] != "M" else "ME")

        self.model=model
        self.freq=data_info['freq']

        super().__init__(
            prediction_length=data_info['h'],
            context_length=data_info['h']*2,
            loss="nll",
            input_size=1,
            num_time_features=len(time_features)+1,
            num_dynamic_real_features=0,
            num_static_categorical_features=1,
            num_static_real_features=0,
            cardinality=[len(datasets['train'])],
            embedding_dimension=[3],
            lags_sequence=lags_sequence,

            # transformer params
            d_model=32,                      
            encoder_layers=4,                      
            decoder_layers=4,             
            encoder_attention_heads=2,             
            decoder_attention_heads=2,
            encoder_ffn_dim=32,                 
            decoder_ffn_dim=32, 
            activation_function="gelu",
            dropout=0.1,
            encoder_layerdrop=0.1,  
            decoder_layerdrop=0.1,      
            attention_dropout=0.1,
            activation_dropout=0.1,
            num_parallel_samples=100,
            init_std=0.02,
            use_cache=True,

            **kwargs  
        )
        
    
    def __str__(self):
        return "<ModelConfig class>"
    
    ### Create Model
    def build_model(self, distribution_head, scaling):
        assert distribution_head in ["poisson","negative_binomial", "tweedie", "tweedie-fix"]

        if self.model == "deepAR":
            if distribution_head == "poisson": self.distribution_output = PoissonOutput()
            if distribution_head == "negative_binomial": self.distribution_output = NegativeBinomialOutput()
            if distribution_head == "tweedie": self.distribution_output = TweedieOutput()
            if distribution_head == "tweedie-fix": self.distribution_output = FixedDispersionTweedieOutput()
        if self.model == "transformer":
            if distribution_head == "poisson": self.distribution_output = "poisson"
            if distribution_head == "negative_binomial": self.distribution_output = "negative_binomial"
            if distribution_head == "tweedie": self.distribution_output = "tweedie"
            if distribution_head == "tweedie-fix": self.distribution_output = 'fixed dispersion tweedie'

        # TODO scaling
        # ...

        if self.model == "deepAR":
            return DeepARModel(
                        freq = self.freq,
                        prediction_length = self.prediction_length,
                        context_length = 2 * self.prediction_length,
                        distr_output = self.distribution_output,
                        lags_seq = self.lags_sequence,
                        num_feat_dynamic_real = self.num_dynamic_real_features + self.num_time_features,
                        num_feat_static_cat = self.num_static_categorical_features,
                        cardinality = self.cardinality,
                        embedding_dimension = self.embedding_dimension,
                        scaling = self.scaling,
                        num_parallel_samples = self.num_parallel_samples)
        if self.model == "transformer":
            return TimeSeriesTransformerForPrediction(self)

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
            num_parallel_samples = config.num_parallel_samples
        ).detach().numpy()
    return(predictions)