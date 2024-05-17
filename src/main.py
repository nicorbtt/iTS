from dataloader import load_raw, create_datasets, create_dataloaders
from visual import learning_curves, Logger
from models import ModelConfigBuilder, forward, predict, EarlyStop
from measures import compute_intermittent_indicators, label_intermittent, quantile_loss_sample, rho_risk_sample

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import json
import torch
from gluonts.dataset.field_names import FieldName
from accelerate import Accelerator
from torch.optim import AdamW
import random


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
    parser.add_argument('--distribution_head', type=str, choices=['poisson','negbin', 'tweedie', 'tweedie-fix', 'tweedie-priors', 'zero-inf-pois'], default='tweedie', help="Specify distribution_head, default is 'tweedie'")
    parser.add_argument('--scaling', type=str, default=None, choices=['mase', 'mean', 'mean-demand', None], help="Specify scaling, default is None")
    parser.add_argument('--model_params', type=json_file_path, default=None, help='Specify the ventual path (.json file) of the model parameters, default is None')
    parser.add_argument('--num_epochs', type=int, default=int(1e4), help='Specify max training epochs, default is 1e4')
    parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size, default is 128')
    parser.add_argument('--silent', '-s', action='store_true', help='Silent, i.e. no verbose')
    parser.add_argument('--log', '-log', action='store_true', help='Whether to save the log')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility, default is 42')
    parser.add_argument('--max_idle_transforms', type=str, default="10000", help='(mini-batch sampling) Maximum number of times a transformation can receive an input without returning an output. This parameter is intended to catch infinite loops or inefficiencies, when transformations never or rarely return something, default is 10000')
    parser.add_argument('--sample_zero_percentage', type=str, default="1", help='(mini-batch sampling) Maximum percentage of 0s allowed for each sample, default is 1 (i.e. do not discard anything)')
    parser.add_argument('--p_reject', type=str, default="1", help='(mini-batch sampling) Probability of discard, default is 1 (i.e. discard all)')
    parser_args = parser.parse_args()

    # Set seed (everywhere)
    random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)
    torch.use_deterministic_algorithms(mode=True)

    # Seting parameters of mini-batch sampling 
    os.environ.setdefault("GLUONTS_MAX_IDLE_TRANSFORMS", parser.max_idle_transforms)
    os.environ.setdefault("iTS_sample_zero_percentage", parser.sample_zero_percentage)
    os.environ.setdefault("iTS_p_sample_zero_percentage_reject", parser.p_reject)

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_folder_name = (
        parser_args.model + "__" +
        parser_args.dataset_name + "__" +
        parser_args.distribution_head + "__" +
        (parser_args.scaling if parser_args.scaling else "none") + "__" +
        dt
    )
    model_folder_path = os.path.join(os.path.expanduser("~/switchdrive"), "iTS", "trained_models", model_folder_name)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    
    stdout = open(os.path.join(model_folder_path, "log.txt"), 'x') if parser_args.log else sys.stdout
    logger = Logger(disable=parser_args.silent, stdout=stdout)
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

    
    # 5. Plot of Learning curves
    learning_curves(history, path=model_folder_path, likelihood=model_builder.distribution_head, scaling=model_builder.scaling)
    # 6. Save the model and params
    torch.save(early_stop.best_model, os.path.join(model_folder_path, "model_state.model"))
    json.dump(model_builder.export_config(), open(os.path.join(model_folder_path, "model_params.json"), "w"))
    json.dump({'datetime': dt, 
                'dataset': parser_args.dataset_name, 
                'model': model_builder.model,
                'distribution_head': model_builder.distribution_head,
                'scaling': model_builder.scaling,
                'epoch': epoch,
                'early_stop': early_stop.stop}, open(os.path.join(model_folder_path, "experiment.json"), "w"))

    # Load the model from disk
    logger.log("Loading the model")
    model_params = json.load(open(os.path.join(model_folder_path, "model_params.json"), "r"))
    model_state = torch.load(os.path.join(model_folder_path, "model_state.model"))
    experiment_info = json.load(open(os.path.join(model_folder_path, "experiment.json"), "r"))
    model_builder = ModelConfigBuilder(model=experiment_info['model'], distribution_head=experiment_info['distribution_head'], scaling=experiment_info['scaling'])
    model_builder.build(data_info, **model_params)
    model = model_builder.get_model()
    model.load_state_dict(model_state)

    # Prediction
    logger.log("Generating forecasts")
    model.eval()
    forecasts_list = []
    for i, batch in enumerate(test_dataloader):
        logger.log("Batch " + str(i+1) + " out of " + str(len(list(test_dataloader))))
        forecasts_list.append(predict(model, batch, device, CONFIG))
        forecasts = np.vstack(forecasts_list)
    #forecasts = np.vstack([predict(model, batch, device, CONFIG) for batch in test_dataloader])
    actuals = np.array([x[FieldName.TARGET][-data_info['h']:] for x in datasets['test']])
    assert actuals.shape[0] == forecasts.shape[0]
    assert actuals.shape[1] == forecasts.shape[2]
    assert forecasts.ndim == 3

    np.save(os.path.join(model_folder_path,"actuals.npy"), actuals)
    np.save(os.path.join(model_folder_path,"forecasts.npy"), forecasts)

    # Quantile Loss
    logger.log("Computing performance measures")
    idx_intermittent = np.logical_and(adi >= 1.32, cv2 < .49)
    idx_intermittent_and_lumpy = adi >= 1.32
    metrics = {'quantile_loss' : {'all' : quantile_loss_sample(actuals, forecasts),
                                   'intermittent' : quantile_loss_sample(actuals[idx_intermittent,:], forecasts[idx_intermittent,:,:]),
                                   'intermittent_and_lumpy' : quantile_loss_sample(actuals[idx_intermittent_and_lumpy,:], forecasts[idx_intermittent_and_lumpy,:,:])},
               'rho_risk_nan' : {'all' : rho_risk_sample(actuals, forecasts, zero_denom = np.nan),
                                 'intermittent' : rho_risk_sample(actuals[idx_intermittent,:], forecasts[idx_intermittent,:,:], zero_denom = np.nan),
                                 'intermittent_and_lumpy' : rho_risk_sample(actuals[idx_intermittent_and_lumpy,:], forecasts[idx_intermittent_and_lumpy,:,:], zero_denom = np.nan)},
               'rho_risk_1' : {'all' : rho_risk_sample(actuals, forecasts, zero_denom = 1.),
                               'intermittent' : rho_risk_sample(actuals[idx_intermittent,:], forecasts[idx_intermittent,:,:], zero_denom = 1.),
                               'intermittent_and_lumpy' : rho_risk_sample(actuals[idx_intermittent_and_lumpy,:], forecasts[idx_intermittent_and_lumpy,:,:], zero_denom = 1.)}}
    json.dump(metrics, open(os.path.join(model_folder_path,"metrics.json"), "w"))

    logger.log(f"End. Find results in {model_folder_path}")
    logger.off()