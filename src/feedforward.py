import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0"

import sys
import numpy as np

from dataloader import load_raw, create_datasets, create_dataloaders
from visual import learning_curves, Logger
from models import EarlyStop
from measures import compute_intermittent_indicators, label_intermittent, quantile_loss, quantile_loss_, brier_score

from gluonts.torch.model.simple_feedforward import SimpleFeedForwardModel
from gluonts.torch.distributions import (
    PoissonOutput,
    NegativeBinomialOutput, 
    TweedieOutput, 
    ZeroInflatedNegativeBinomialOutput
)
from gluonts.dataset.field_names import FieldName

import argparse
from datetime import datetime
import json
import random
import torch
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
    parser.add_argument('--lag', type=int, required=True, help="Specify lag")
    parser.add_argument('--distribution_head', type=str, choices=['poisson','negbin', 'tweedie', 'zinb', 'zero-inf-pois'], default='tweedie', help="Specify distribution_head, default is 'tweedie'")
    parser.add_argument('--scaling', type=str, default=None, choices=['mean', 'mean-demand', None], help="Specify scaling, default is None")
    parser.add_argument('--model_params', type=json_file_path, default=None, help='Specify the ventual path (.json file) of the model parameters, default is None')
    parser.add_argument('--num_epochs', type=int, default=int(1e4), help='Specify max training epochs, default is 1e4')
    parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size, default is 128')
    parser.add_argument('--silent', '-s', action='store_true', help='Silent, i.e. no verbose')
    parser.add_argument('--log', '-log', action='store_true', help='Whether to save the log')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility, default is 42')
    parser.add_argument('--cpu', type=bool, default=False, help='Select the device')
    parser.add_argument('--max_idle_transforms', type=str, default="10000", help='(mini-batch sampling) Maximum number of times a transformation can receive an input without returning an output. This parameter is intended to catch infinite loops or inefficiencies, when transformations never or rarely return something, default is 10000')
    parser.add_argument('--sample_zero_percentage', type=str, default="1", help='(mini-batch sampling) Maximum percentage of 0s allowed for each sample, default is 1 (i.e. do not discard anything)')
    parser.add_argument('--p_reject', type=str, default="1", help='(mini-batch sampling) Probability of discard, default is 1 (i.e. discard all)')
    parser_args = parser.parse_args()

    # Set seed (everywhere)
    random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)
    torch.use_deterministic_algorithms(mode=parser_args.cpu)

    # Seting parameters of mini-batch sampling 
    os.environ.setdefault("GLUONTS_MAX_IDLE_TRANSFORMS", parser_args.max_idle_transforms)
    os.environ.setdefault("iTS_sample_zero_percentage", parser_args.sample_zero_percentage)
    os.environ.setdefault("iTS_p_sample_zero_percentage_reject", parser_args.p_reject)

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_folder_name = (
        "feedforward_l" + str(parser_args.lag) + "__" +
        parser_args.dataset_name + "__" +
        parser_args.distribution_head + "__" +
        ("mean-demand" if parser_args.scaling else "none") + "__" +
        dt
    )
    model_folder_path = os.path.join(os.getcwd(), "trained_models", model_folder_name)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    
    stdout = open(os.path.join(model_folder_path, "log.txt"), 'x') if parser_args.log else sys.stdout
    logger = Logger(disable=parser_args.silent, stdout=stdout)
    logger.log(f"Random seed={parser_args.seed}")
    # Import data
    logger.log(f"Loading dataset {parser_args.dataset_name}")
    data_raw, data_info = load_raw(dataset_name=parser_args.dataset_name, datasets_folder_path=os.path.join("data"))

    # Compute intermittent indicators
    logger.log(f"Computing intermittent indicators")
    adi, cv2 = compute_intermittent_indicators(data_raw, data_info['h'])
    data_info['intermittent'] = label_intermittent(adi, cv2, f="intermittent")
    data_info['lumpy'] = label_intermittent(adi, cv2, f="lumpy")

    # Create Datasets (train, valid, test) objects
    logger.log("Preparing dataset")
    datasets = create_datasets(data_raw, data_info)
    config = {k:0 for k in ['num_feat_static_cat' , 'num_feat_static_real', 'num_feat_dynamic_real']}
    config["prediction_length"] = data_info['h']
    config["context_length"] = parser_args.lag
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(config, datasets, data_info, batch_size=parser_args.batch_size)

    logger.log(f"Building the model")
    def ff_builder(distribution_head, scaling, prediction_length, context_length):
        if distribution_head == "tweedie":
            distr_output = TweedieOutput()
        elif distribution_head == "negbin":
            distr_output = NegativeBinomialOutput()
        elif distribution_head == "zinb":
            distr_output = ZeroInflatedNegativeBinomialOutput()
        elif distribution_head == "poisson":
            distr_output = PoissonOutput()
        else:
            raise ValueError(f"Distribution head {distribution_head} not found")
        return SimpleFeedForwardModel(
            scale = scaling,
            prediction_length = prediction_length,
            context_length = context_length,
            hidden_dimensions= [32, 32, 32, 32, 32], 
            distr_output = distr_output,
            batch_norm = False,
        )
    
    model = ff_builder(parser_args.distribution_head, parser_args.scaling, data_info['h'], parser_args.lag)
    accelerator = Accelerator(cpu=parser_args.cpu)
    device = accelerator.device
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-8)
    #model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader)
    early_stop = EarlyStop(logger, patience=20, min_delta=1e-3)

    # Training loop
    history = { 'train_loss': [], 'val_loss': []}
    logger.log(f'Training on device={device}')
    for epoch in range(parser_args.num_epochs): 
        # 1. Training
        train_loss = 0
        model.train()
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.loss(
                past_target = batch['past_values'].to(device),
                future_target = batch['future_values'].to(device),
                future_observed_values = batch['future_observed_mask'].to(device),
            ).mean()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
        history['train_loss'].append(train_loss / idx)
        # 2. Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                loss = model.loss(
                    past_target = batch['past_values'].to(device),
                    future_target = batch['future_values'].to(device),
                    future_observed_values = batch['future_observed_mask'].to(device),
                ).mean()
                val_loss += loss.item()
        history['val_loss'].append(val_loss / idx)
        logger.log_epoch(epoch, history)
        # 3. Early stopping
        early_stop.update(model, epoch, history['val_loss'][-1])
        if early_stop.stop: break

    # 4. Save informations and modelss
    torch.save(early_stop.best_model, os.path.join(model_folder_path, "model_state.model"))
    json.dump({"scaling" : parser_args.scaling,
               "prediction_length" : data_info["h"],
               "context_length" : parser_args.lag,
               "distribution_head" : parser_args.distribution_head}, open(os.path.join(model_folder_path, "model_params.json"), "w"))
    json.dump({'datetime': dt, 
                'dataset': parser_args.dataset_name, 
                'lag' : parser_args.lag,
                'distribution_head': parser_args.distribution_head,
                'scaling': parser_args.scaling,
                'epoch': epoch,
                'early_stop': early_stop.stop,
                'seed':parser_args.seed}, open(os.path.join(model_folder_path, "experiment.json"), "w"))

    # Load the model from disk
    logger.log("Loading the model")
    model_params = json.load(open(os.path.join(model_folder_path, "model_params.json"), "r"))
    model_state = torch.load(os.path.join(model_folder_path, "model_state.model"))
    experiment_info = json.load(open(os.path.join(model_folder_path, "experiment.json"), "r"))
    model = ff_builder(**model_params)
    model.load_state_dict(model_state)

    # Prediction
    accelerator = Accelerator(cpu=parser_args.cpu)
    device = accelerator.device
    model.to(device)

    logger.log("Generating forecasts, device="+str(device))
    model.eval()
    quantile_forecasts_list, probs0_list = [], []
    actuals = np.array([x[FieldName.TARGET][-data_info['h']:] for x in datasets['test']])
    for i, batch in enumerate(test_dataloader):
        logger.log("Batch " + str(i+1) + " out of " + str(len(list(test_dataloader))))
        with torch.no_grad():
            distr_args, loc, scale = model(batch['past_values'].to(device))
            distribution = model.distr_output.distribution(distr_args, loc=loc, scale=scale)
            samples = distribution.sample(torch.Size([10000])).cpu()
            quantile_forecasts_list.append(
                torch.quantile(samples, torch.tensor([0.5, 0.8, 0.9, 0.95, 0.99]), axis=0).permute(1,2,0).numpy()
                )
            probs0_list.append(
                (samples == 0).to(torch.float32).mean(axis=0).numpy()
                )
    quantile_forecasts = np.concatenate(quantile_forecasts_list, axis=0)
    probs0 = np.concatenate(probs0_list, axis=0)
    assert actuals.shape[0] == quantile_forecasts.shape[0]
    assert actuals.shape[1] == quantile_forecasts.shape[1]
    assert quantile_forecasts.ndim == 3
    assert actuals.shape == probs0.shape 
    for k, q in enumerate([0.5, 0.8, 0.9, 0.95, 0.99]):
        ql = quantile_loss_(actuals, quantile_forecasts[:,:,k], q, avg=False).mean(axis=1)
        np.save(os.path.join(model_folder_path,"ql_"+str(q)+".npy"), ql)
    if parser_args.lag == 1:
        np.save(os.path.join(model_folder_path,"actuals.npy"), actuals)

    # Quantile Loss
    logger.log("Computing performance measures")
    idx_intermittent = np.logical_and(adi >= 1.32, cv2 < .49)
    idx_intermittent_and_lumpy = adi >= 1.32
    metrics = {'quantile_loss' : {'all' : quantile_loss(actuals, quantile_forecasts),
                                   'intermittent' : quantile_loss(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:]),
                                   'intermittent_and_lumpy' : quantile_loss(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:])},
               'brier_score' : {'all' : brier_score(actuals, probs0),
                                   'intermittent' : brier_score(actuals[idx_intermittent,:], probs0[idx_intermittent,:]),
                                   'intermittent_and_lumpy' : brier_score(actuals[idx_intermittent_and_lumpy,:], probs0[idx_intermittent_and_lumpy,:])}
    }
    json.dump(metrics, open(os.path.join(model_folder_path,"metrics.json"), "w"))
    logger.log(f"End. Find results in {model_folder_path}")
    logger.off()
            