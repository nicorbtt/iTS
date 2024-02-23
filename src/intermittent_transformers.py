import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW

from accelerate import Accelerator

import transformers
from transformers.models.time_series_transformer import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

config = TimeSeriesTransformerConfig(

    prediction_length=prediction_length,
    context_length=prediction_length*2,
    lags_sequence=lags_sequence,
    num_time_features= 2, # month of year + "age"
    num_static_categorical_features=1,
    cardinality=[len(train_dataset)],
    embedding_dimension=[2],
    scaling=None,

    encoder_layers=4, 
    decoder_layers=4, 
    d_model=32,

    num_parallel_samples=1000,
    distribution_output="tweedie",
)

model = TimeSeriesTransformerForPrediction(config)

accelerator = Accelerator()
device = accelerator.device

model.to(device)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
)

for epoch in range(60):
    train_running_loss = 0

    model.train()
    for idx, batch in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device),
            static_real_features=None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        )
        loss = outputs.loss
        train_running_loss += loss.item()

        accelerator.backward(loss)
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
