from dataloader import load_raw, create_datasets, create_dataloaders
from visual import learning_curves, forecast_plot, Logger
from models import ModelConfigBuilder, forward, predict, EarlyStop
from measures import compute_intermittent_indicators, label_intermittent, quantile_loss

import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import json
import math
import torch
from gluonts.dataset.field_names import FieldName
from accelerate import Accelerator
from torch.optim import AdamW


if __name__ == "__main__":
    # Command line parser
    def json_file_path(model_params):
        if not os.path.isfile(model_params):
            raise argparse.ArgumentTypeError(f"{model_params} is not a valid file path")
        if not model_params.lower().endswith('.json'):
            raise argparse.ArgumentTypeError("File must have a .json extension")
        return model_params
    parser = argparse.ArgumentParser(description="iTS")
    parser.add_argument('--dataset_name', type=str, choices=['OnlineRetail', 'Auto', 'RAF', 'carparts', 'syph', 'M5'], required=True, help='Specify dataset name')
    parser.add_argument('--model', type=str, choices=['deepAR','transformer'], required=True, help="Specify model")
    parser.add_argument('--distribution_head', type=str, choices=['poisson','negbin', 'tweedie', 'tweedie-fix'], default='tweedie', help="Specify distribution_head, default is 'tweedie'")
    parser.add_argument('--scaling', type=str, default=None, choices=['mase', 'mean', 'mean-demand', None], help="Specify scaling, default is None")
    parser.add_argument('--model_params', type=json_file_path, default=None, help='Specify the ventual path (.json file) of the model parameters, default is None')
    parser.add_argument('--num_epochs', type=int, default=int(1e4), help='Specify max training epochs, default is 1e4')
    parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size, default is 128')
    parser.add_argument('--silent', '-s', action='store_true', help='Silent, i.e. no verbose')
    parser_args = parser.parse_args()

    logger = Logger(disable=parser_args.silent)
    # Import data
    logger.log(f"Loading dataset {parser_args.dataset_name}")
    data_raw, data_info = load_raw(dataset_name=parser_args.dataset_name, datasets_folder_path=os.path.join("data"))

    # Compute intermittent indicators
    logger.log(f"Computing intermittent indicators")
    adi, cv2 = compute_intermittent_indicators(data_raw)
    data_info['intermittent'] = label_intermittent(adi, cv2, f="intermittent")
    data_info['lumpy'] = label_intermittent(adi, cv2, f="lumpy")

    # Create Datasets (train, valid, test) objects
    datasets = create_datasets(data_raw, data_info)

    # Model config
    model_builder = ModelConfigBuilder(model=parser_args.model, distribution_head=parser_args.distribution_head, scaling=parser_args.scaling)
    loaded_params = json.load(open(parser_args.model_params)) if parser_args.model_params else {}
    model_builder.build(data_info, **loaded_params)
    CONFIG = model_builder.params

    # Dataloaders
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(CONFIG, datasets, data_info, batch_size=parser_args.batch_size)

    # Build the model
    logger.log(f"Building the model")
    model = model_builder.get_model()

    # Training setup
    accelerator = Accelerator(cpu=True)
    device = accelerator.device
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader)
    early_stop = EarlyStop(logger, patience=20, min_delta=1e-3)

    # Training loop
    history = { 'train_loss': [], 'val_loss': []}
    logger.log(f'Training on device={device}')
    for epoch in range(parser_args.num_epochs):
        # 1. Training
        train_loss = 0.0
        model.train()
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = forward(model, batch, device, CONFIG)
            train_loss += loss.item()
            accelerator.backward(loss); optimizer.step()
        history['train_loss'].append(train_loss / idx)
        # 2. Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                loss = forward(model, batch, device, CONFIG)
                val_loss += loss.item()
        history['val_loss'].append(val_loss / idx)
        logger.log_epoch(epoch, history)
        # 3. Early Stopping
        early_stop.update(model, epoch, history['val_loss'][-1])
        if early_stop.stop: break

    dt = datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
    model_folder_name = model_builder.model + "__" + parser_args.dataset_name + "__" + dt
    model_folder_path = os.path.join("trained_models", model_folder_name)
    if not os.path.exists(path=model_folder_path):
        os.makedirs(model_folder_path)
        # 5. Plot of Learning curves
        learning_curves(history, path=model_folder_path)
        # 6. Save the model and params
        torch.save(early_stop.best_model, os.path.join("trained_models", model_folder_name, "model_state.model"))
        json.dump(model_builder.export_config(), open(os.path.join("trained_models",model_folder_name,"model_params.json"), "w"))
        json.dump({'datetime': dt, 
                'dataset': parser_args.dataset_name, 
                'model': model_builder.model,
                'distribution_head': model_builder.distribution_head,
                'scaling': model_builder.scaling,
                'epoch': epoch,
                'early_stop': early_stop.stop}, open(os.path.join("trained_models",model_folder_name,"experiment.json"), "w"))

    # Load the model from disk
    logger.log("Loading the model")
    model_params = json.load(open(os.path.join("trained_models",model_folder_name,"model_params.json"), "r"))
    model_state = torch.load(os.path.join(model_folder_path,"model_state.model"))
    experiment_info = json.load(open(os.path.join("trained_models",model_folder_name,"experiment.json"), "r"))
    model_builder = ModelConfigBuilder(model=experiment_info['model'], distribution_head=experiment_info['distribution_head'], scaling=experiment_info['scaling'])
    model_builder.build(data_info, **model_params)
    model = model_builder.get_model()
    model.load_state_dict(model_state)

    # Prediction
    logger.log("Generating forecasts")
    model.eval()
    if parser_args.silent:
        forecasts = [predict(model, batch, device, CONFIG) for batch in test_dataloader]
    else:
        forecasts = [predict(model, batch, device, CONFIG) 
                    for batch in tqdm(test_dataloader, total=math.ceil(len(datasets['test']) / parser_args.batch_size))]
    forecasts = np.vstack(forecasts)
    actuals = np.array([x[FieldName.TARGET][-data_info['h']:] for x in datasets['test']])
    assert actuals.shape[0] == forecasts.shape[0]
    assert actuals.shape[1] == forecasts.shape[2]
    assert forecasts.ndim == 3

    np.save(os.path.join(model_folder_path,"actuals.npy"), actuals)
    np.save(os.path.join(model_folder_path,"forecasts.npy"), forecasts)

    # Quantile Loss
    logger.log("Computing performance measures")
    metrics = {}
    metrics['quantile_loss'] = quantile_loss(actuals, forecasts, q=[0.25, 0.5, 0.8, 0.9, 0.95, 0.99])
    json.dump(metrics, open(os.path.join(model_folder_path,"metrics.json"), "w"))

    logger.log(f"End. Find results in {model_folder_path}")