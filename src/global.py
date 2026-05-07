import os
import sys
import argparse
from datetime import datetime
import numpy as np
import json
import torch

if not os.environ.get("IS_DOCKER") == "1":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0"

from dataloader import load_raw, create_datasets, create_dataloaders, create_tabular
from visual import learning_curves, Logger
from models import ModelConfigBuilder, forward, predict, EarlyStop, ParamSampler
from measures import compute_intermittent_indicators, quantile_loss, quantile_loss_scaled_in_sample, rmsse, quantile_loss_scaled_mae

from torch.optim import AdamW
from accelerate import Accelerator
import lightgbm as lgb
import random

if __name__ == "__main__":
    # Command line parser
    def json_file_path(model_params):
        if not os.path.isfile(model_params):
            raise argparse.ArgumentTypeError(f"{model_params} is not a valid file path")
        if not model_params.lower().endswith('.json'):
            raise argparse.ArgumentTypeError("File must have a .json extension")
        return model_params
    parser = argparse.ArgumentParser(description="iLocGlob: global vs local models for intermittent time series forecasting")
    parser.add_argument('--dataset_name', '--dataset-name', dest='dataset_name', type=str, choices=['OnlineRetail', 'Auto', 'RAF', 'carparts', 'syph', 'M5', 'crime', 'VN1', 'UCI'], required=True, help='Specify dataset name')
    parser.add_argument('--model', type=str, choices=['deepAR','transformer','informer', 'autoformer', 'patchTST', 'tide', 'feedforward', 'dlinear', 'lightgbm'], required=True, help="Specify model")
    parser.add_argument('--distribution_head', type=str, choices=['poisson','negbin', 'tweedie', 'hsnb', 'quantile', 'isqf', 'iqn'], default='tweedie', help="Specify distribution_head, default is 'tweedie'")
    parser.add_argument('--scaling', type=str, default=None, choices=['mase', 'mean', 'mean-demand', None], help="Specify scaling, default is None")
    parser.add_argument('--model_params', type=json_file_path, default=None, help='Specify the ventual path (.json file) of the model parameters, default is None')
    parser.add_argument('--num_epochs', type=int, default=int(1e4), help='Specify max training epochs, default is 1e4')
    parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size, default is 128')
    parser.add_argument('--silent', '-s', action='store_true', help='Silent, i.e. no verbose')
    parser.add_argument('--log', '-log', action='store_true', help='Whether to save the log')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility, default is 42')
    parser.add_argument('--cpu', type=bool, default=False, help='Select the device')
    parser.add_argument('--zero_shot_training_dataset', type=str, default=None, help='Specify a different dataset for training')
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
        parser_args.model + "__" +
        parser_args.dataset_name + "__" +
        parser_args.distribution_head + "__" +
        (parser_args.scaling if parser_args.scaling else "none") + "__" +
        dt
    )

    trained_models_dir = "/opt/trained_models" if os.environ.get("IS_DOCKER") == "1" else os.path.join(os.getcwd(), "trained_models")
    model_folder_path = os.path.join(trained_models_dir, model_folder_name)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
        os.makedirs(os.path.join(model_folder_path, "forecasts"))
    
    stdout = open(os.path.join(model_folder_path, "log.txt"), 'x') if parser_args.log else sys.stdout
    logger = Logger(disable=parser_args.silent, stdout=stdout)
    logger.log(f"Random seed={parser_args.seed}")
    # Import data
    logger.log(f"Loading dataset {parser_args.dataset_name}")
    data_raw, data_info = load_raw(dataset_name=parser_args.dataset_name, datasets_folder_path=os.path.join("data"))

    # Compute intermittent indicators
    logger.log(f"Computing intermittent indicators")
    adi, cv2 = compute_intermittent_indicators(data_raw, data_info['h'])
    # data_info['intermittent'] = label_intermittent(adi, cv2, f="intermittent")
    # data_info['lumpy'] = label_intermittent(adi, cv2, f="lumpy")

    # If required, create the additional dataset
    # if parser_args.zero_shot_training_dataset is not None:
    #     logger.log(f"Loading training dataset {parser_args.zero_shot_training_dataset}")
    #     train_data_raw, train_data_info = load_raw(dataset_name=parser_args.zero_shot_training_dataset, datasets_folder_path=os.path.join("data"))
    #     assert train_data_info['freq'] == data_info['freq']
    #     train_data_info['h'], train_data_info['w'] = data_info['h'], data_info['w']
    #     train_datasets = create_datasets(train_data_raw, train_data_info, zero_id = True)
        
    # Create Datasets (train, valid, test) objects
    datasets = create_datasets(data_raw, data_info) #zero_id = if parser_args.zero_shot_training_dataset is None else True)

    # Model config
    model_builder = ModelConfigBuilder(model=parser_args.model, distribution_head=parser_args.distribution_head, scaling=parser_args.scaling)
    loaded_params = json.load(open(parser_args.model_params)) if parser_args.model_params else {}
    model_builder.build(data_info, **loaded_params)
    CONFIG = model_builder.params

    if parser_args.model == "lightgbm":

        logger.log(f"Creating tabular datasets")
        train_tabular, valid_tabular, test_tabular = create_tabular(CONFIG, datasets, data_info)

        logger.log(f"Building the model")
        model = model_builder.get_model()
        param_sampler = ParamSampler()

        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(parser_args.num_epochs):
            params = param_sampler.sample()

            X_train, y_train = train_tabular
            try:
                model.train(params,
                        lgb.Dataset(X_train, label=y_train, categorical_feature=[0]))
                #loss = forward(model, train_tabular, "cpu", config=CONFIG)
                val_loss = []
                for i, tab in enumerate(valid_tabular):
                    logger.log("Timestep " + str(i+1) + " out of " + str(data_info['h']))
                    loss = forward(model, tab, "cpu", config=CONFIG)
                    val_loss.append(loss)
                history['train_loss'].append(None)
                history['val_loss'].append(np.mean(val_loss))
                param_sampler.update(np.mean(val_loss), params, model, iter_num=epoch)
                logger.log_iter(epoch, params, np.mean(val_loss), param_sampler.best_iter)

            except Exception as e:
                logger.log(f"Error occurred at iteration {epoch}: {e}")

        # Save the best LightGBMLSS model's booster
        if param_sampler.best_model is not None and hasattr(param_sampler.best_model, 'booster'):# and param_sampler.best_model.booster is not None:
            param_sampler.best_model.booster.save_model(os.path.join(model_folder_path, "model_state.txt"))
        best_val_loss, stop = param_sampler.best_val_loss, False
    
    else:
        logger.log(f"Creating dataloaders")
        train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(CONFIG, datasets, data_info, batch_size=parser_args.batch_size)

        logger.log(f"Building the model")
        model = model_builder.get_model()

        # Training setup
        accelerator = Accelerator(cpu=parser_args.cpu)
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
        best_val_loss, stop = early_stop.best_val_loss, early_stop.stop


    json.dump(model_builder.export_config(), open(os.path.join(model_folder_path, "model_params.json"), "w"))
    json.dump({'datetime': dt, 
               'dataset': parser_args.dataset_name, 
                'model': model_builder.model,
                'distribution_head': model_builder.distribution_head,
                'scaling': model_builder.scaling,
                'epoch': epoch,
                'early_stop': stop,
                'validation_best': best_val_loss,
                'seed':parser_args.seed}, open(os.path.join(model_folder_path, "experiment.json"), "w"), indent=4)


    # Load the model from disk
    logger.log("Loading the model")
    model_params = json.load(open(os.path.join(model_folder_path, "model_params.json"), "r"))
    experiment_info = json.load(open(os.path.join(model_folder_path, "experiment.json"), "r"))
    model_builder = ModelConfigBuilder(model=experiment_info['model'], distribution_head=experiment_info['distribution_head'], scaling=experiment_info['scaling'])
    model_builder.build(data_info, **model_params)
    model = model_builder.get_model()

    # For LightGBMLSS, load the booster if present
    if experiment_info['model'] == 'lightgbm':
        model.booster = lgb.Booster(model_file=os.path.join(model_folder_path, "model_state.txt"))
    else:
        # Deep models: load state dict as before
        model_state = torch.load(os.path.join(model_folder_path, "model_state.model"))
        model.load_state_dict(model_state)

    logger.log("Generating forecasts, device="+(str(device) if experiment_info['model'] != 'lightgbm' else "cpu"))
    qlevels = [0.5, 0.8, 0.9, 0.95, 0.99]
    mean_forecasts, quantile_forecasts = [], [] 

    if model_builder.model == "lightgbm":
        model = param_sampler.best_model # TODO: load the best model from saved version...
        for i, tab in enumerate(test_tabular):
            logger.log("Timestep " + str(i+1) + " out of " + str(data_info['h']))
            X_test, _ = tab
            forecast_samples = model.predict(X_test, pred_type="samples", n_samples=10000).values
            mean_forecasts.append(forecast_samples.mean(axis=1))
            quantile_forecasts.append(np.quantile(forecast_samples, qlevels, axis=1))
        mean_forecasts = np.vstack(mean_forecasts).T
        quantile_forecasts = np.stack(quantile_forecasts, axis=1).swapaxes(0, 2)

    else:
        model.eval()
        accelerator = Accelerator(cpu=parser_args.cpu)
        device = accelerator.device
        model.to(device)

        for i, batch in enumerate(test_dataloader):
            logger.log("Batch " + str(i+1) + " out of " + str(len(list(test_dataloader))))
            forecast_samples = predict(model, batch, device, CONFIG)
            mean_forecasts.append(forecast_samples.mean(axis=1))
            quantile_forecasts.append(np.quantile(forecast_samples, qlevels, axis=1).transpose(1,2,0))
        mean_forecasts = np.vstack(mean_forecasts)
        quantile_forecasts = np.vstack(quantile_forecasts)


    actuals = np.array([x["target"][-data_info['h']:] for x in datasets['test']])
    insample = np.array([x["target"][:-data_info['h']] for x in datasets['test']])
    assert quantile_forecasts.shape == (*actuals.shape, len(qlevels))
    assert mean_forecasts.shape == actuals.shape
    np.save(os.path.join(model_folder_path, os.path.join("forecasts", "mean_forecasts.npy")), mean_forecasts)
    for k, q in enumerate(qlevels):
        np.save(os.path.join(model_folder_path, os.path.join("forecasts", "q"+str(q)+".npy")), quantile_forecasts[:,:,k])
    np.save(os.path.join(model_folder_path, os.path.join("forecasts", "actuals.npy")), actuals)

    # Quantile Loss
    logger.log("Computing performance measures")
    idx_intermittent = np.logical_and(adi >= 1.32, cv2 < .49)
    idx_intermittent_and_lumpy = adi >= 1.32
    idx_non_smooth = adi > 1.
    metrics = {
        'quantile_loss' : {
            'all' : quantile_loss(actuals, quantile_forecasts),
            'intermittent' : quantile_loss(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:]),
            'intermittent_and_lumpy' : quantile_loss(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:]),
            'non-smooth': quantile_loss(actuals[idx_non_smooth,:], quantile_forecasts[idx_non_smooth,:,:])
            },
        'quantile_loss_scaled_in_sample': {
            'all' : quantile_loss_scaled_in_sample(actuals, quantile_forecasts, insample),
            'intermittent' : quantile_loss_scaled_in_sample(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:], insample[idx_intermittent,:]),
            'intermittent_and_lumpy' : quantile_loss_scaled_in_sample(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:], insample[idx_intermittent_and_lumpy,:]),
            'non-smooth': quantile_loss_scaled_in_sample(actuals[idx_non_smooth,:], quantile_forecasts[idx_non_smooth,:,:], insample[idx_non_smooth,:])
            },  
        'quantile_loss_scaled_mae': {
            'all' : quantile_loss_scaled_mae(actuals, quantile_forecasts, insample),
            'intermittent' : quantile_loss_scaled_mae(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:], insample[idx_intermittent,:]),
            'intermittent_and_lumpy' : quantile_loss_scaled_mae(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:], insample[idx_intermittent_and_lumpy,:]),
            'non-smooth': quantile_loss_scaled_mae(actuals[idx_non_smooth,:], quantile_forecasts[idx_non_smooth,:,:], insample[idx_non_smooth,:])
            },
        'rmsse': {
            'all' : rmsse(actuals, mean_forecasts, insample),   
            'intermittent' : rmsse(actuals[idx_intermittent,:], mean_forecasts[idx_intermittent,:], insample[idx_intermittent,:]),
            'intermittent_and_lumpy' : rmsse(actuals[idx_intermittent_and_lumpy,:], mean_forecasts[idx_intermittent_and_lumpy,:], insample[idx_intermittent_and_lumpy,:]),
            'non-smooth': rmsse(actuals[idx_non_smooth,:], mean_forecasts[idx_non_smooth,:], insample[idx_non_smooth,:])
            }
        }
    json.dump(metrics, open(os.path.join(model_folder_path,"metrics.json"), "w"), indent=4)
    logger.log(f"End. Find results in {model_folder_path}")
    logger.off()

    # to debug:
    # --dataset_name carparts --model lightgbm --distribution_head negbin --num_epochs 4
    # --dataset_name carparts --model autoformer --distribution_head negbin --scaling mean-demand --cpu True --num_epochs 3