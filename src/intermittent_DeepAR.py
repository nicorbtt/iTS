import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW

import gluonts
from gluonts.torch.model.deepar.module import DeepARModel
from gluonts.torch.scaler import MASEScaler, MeanDemandScaler
from gluonts.torch.distributions import PoissonOutput, NegativeBinomialOutput, StudentTOutput, TweedieOutput, FixedDispersionTweedieOutput


model = DeepARModel(
    freq = '1M',
    prediction_length = prediction_length,
    context_length = 2*prediction_length,
    distr_output= TweedieOutput(),
    
    lags_seq=lags_sequence,
    num_feat_dynamic_real= 2, # month of year + "age"
    num_feat_static_cat=1,
    cardinality=[len(train_dataset)],
    embedding_dimension=[2],
    scaling='mean demand',

    num_parallel_samples=100
)


optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

for epoch in range(40):
    train_running_loss = 0

    model.train()
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = model.loss(
            feat_static_cat = batch["static_categorical_features"],
            feat_static_real= torch.zeros((batch['past_values'].shape[0],1)),
            past_time_feat = batch["past_time_features"],
            past_target = batch['past_values'],
            past_observed_values = batch['past_observed_mask'],
            future_time_feat = batch['future_time_features'],
            future_target = batch['future_values'],
            future_observed_values = batch["future_observed_mask"],
            ).mean()
        loss.backward()
        train_running_loss += loss.item()

        optimizer.step()

        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1}: Train loss: {train_running_loss}")

model.eval()

forecasts = []

for batch in tqdm(test_dataloader):
    pred = model.forward(
        feat_static_cat = batch["static_categorical_features"],
        feat_static_real= torch.zeros((batch['past_values'].shape[0],1)),
        past_time_feat = batch["past_time_features"],
        past_target = batch['past_values'],
        past_observed_values = batch['past_observed_mask'],
        future_time_feat = batch['future_time_features'],
        num_parallel_samples = 100
        )
    forecasts.append(pred.detach().numpy())

forecasts = np.vstack(forecasts)
