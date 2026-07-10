import argparse
import os
import sys
import numpy as np
from datetime import datetime
import json

from dataloader import load_raw, create_datasets
from visual import Logger
from models import FoundationModel
from measures import compute_intermittent_indicators, quantile_loss_scaled_mae, quantile_loss, quantile_loss_scaled_in_sample, rmsse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iLocGlob: time series foundation models (zero-shot) for intermittent time series forecasting")
    parser.add_argument('--dataset_name', type=str, choices=['OnlineRetail', 'Auto', 'RAF', 'carparts', 'syph', 'M5', 'crime', 'VN1', 'UCI'], required=True, help='Specify dataset name')
    parser.add_argument('--model', type=str, choices=['chronos2', 'toto', 'toto2', 'timesfm', 'tirex'], required=True, help="Specify foundation model")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of sample paths drawn for models relying on sampling (toto), default is 100')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of series forecast per model call, default is 32')
    parser.add_argument('--device', type=str, default="cpu", help='Device used to run the pretrained model, default is cpu')
    parser.add_argument('--online', action='store_true', help='Allow network access to the Hugging Face Hub, needed to download a checkpoint the first time. Default is offline, i.e. use only the local cache')
    parser.add_argument('--silent', '-s', action='store_true', help='Silent, i.e. no verbose')
    parser.add_argument('--log', '-log', action='store_true', help='Whether to save the log')
    parser_args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "0" if parser_args.online else "1"

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_folder_name = (
        parser_args.model + "__" +
        parser_args.dataset_name + "__" +
        dt
    )
    trained_models_dir = "/opt/trained_models" if os.environ.get("IS_DOCKER") == "1" else os.path.join(os.getcwd(), "trained_models")
    model_folder_path = os.path.join(trained_models_dir, model_folder_name)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
        os.makedirs(os.path.join(model_folder_path, "forecasts"))

    stdout = open(os.path.join(model_folder_path, "log.txt"), 'x') if parser_args.log else sys.stdout
    logger = Logger(disable=parser_args.silent, stdout=stdout)
    # Import data
    logger.log(f"Loading dataset {parser_args.dataset_name}")
    data_raw, data_info = load_raw(dataset_name=parser_args.dataset_name, datasets_folder_path=os.path.join("data"))

    # Compute intermittent indicators
    logger.log(f"Computing intermittent indicators")
    adi, cv2 = compute_intermittent_indicators(data_raw, data_info['h'])

    # create datasets and load the pretrained foundation model (zero-shot, no training)
    datasets = create_datasets(data_raw, data_info)
    logger.log(f"Loading pretrained model {parser_args.model}")
    model = FoundationModel(parser_args.model, data_info,
                            num_samples=parser_args.num_samples,
                            device=parser_args.device)

    logger.log("Generate forecasts")
    qlevels = [0.5, 0.8, 0.9, 0.95, 0.99]
    valid_targets = datasets['valid'][:]['target']
    num_batches = (len(valid_targets) - 1) // parser_args.batch_size + 1
    mean_forecasts_list, quantile_forecasts_list = [], []
    for b in range(num_batches):
        logger.log("Batch " + str(b+1) + " out of " + str(num_batches))
        batch = valid_targets[b*parser_args.batch_size : (b+1)*parser_args.batch_size]
        mean_forecast, quantile_forecast = model.forecast(batch)
        mean_forecasts_list.append(mean_forecast)
        quantile_forecasts_list.append(quantile_forecast)
    mean_forecasts = np.concatenate(mean_forecasts_list, axis=0)
    quantile_forecasts = np.concatenate(quantile_forecasts_list, axis=0)

    actuals = np.array([x["target"][-data_info['h']:] for x in datasets['test']])
    insample = np.array([x["target"][:-data_info['h']] for x in datasets['test']])
    assert quantile_forecasts.shape == (*actuals.shape, len(qlevels))
    assert mean_forecasts.shape == actuals.shape
    os.makedirs(os.path.join(model_folder_path, "forecasts"), exist_ok=True)
    np.save(os.path.join(model_folder_path, os.path.join("forecasts", "mean_forecasts.npy")), mean_forecasts)
    for k, q in enumerate(qlevels):
        np.save(os.path.join(model_folder_path, os.path.join("forecasts", "q"+str(q)+".npy")), quantile_forecasts[:,:,k])
    np.save(os.path.join(model_folder_path, os.path.join("forecasts", "actuals.npy")), actuals)
    json.dump({'datetime': dt,
                'dataset': parser_args.dataset_name,
                'model': parser_args.model,
                'num_samples': parser_args.num_samples,
                'batch_size': parser_args.batch_size,
                'device': parser_args.device}, open(os.path.join(model_folder_path, "experiment.json"), "w"), indent=4)


    logger.log("Computing performance measures")
    idx_intermittent = np.logical_and(adi >= 1.32, cv2 < .49)
    idx_intermittent_and_lumpy = adi >= 1.32
    non_smooth = adi > 1.
    metrics = {
        'quantile_loss' : {
            'all' : quantile_loss(actuals, quantile_forecasts),
            'intermittent' : quantile_loss(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:]),
            'intermittent_and_lumpy' : quantile_loss(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:]),
            'non-smooth' : quantile_loss(actuals[non_smooth,:], quantile_forecasts[non_smooth,:,:])
            },
        'quantile_loss_scaled_in_sample' : {
            'all' : quantile_loss_scaled_in_sample(actuals, quantile_forecasts, insample),
            'intermittent' : quantile_loss_scaled_in_sample(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:], insample[idx_intermittent,:]),
            'intermittent_and_lumpy' : quantile_loss_scaled_in_sample(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:], insample[idx_intermittent_and_lumpy,:]),
            'non-smooth' : quantile_loss_scaled_in_sample(actuals[non_smooth,:], quantile_forecasts[non_smooth,:,:], insample[non_smooth,:])
            },
        'quantile_loss_scaled_mae' : {
            'all' : quantile_loss_scaled_mae(actuals, quantile_forecasts, insample),
            'intermittent' : quantile_loss_scaled_mae(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:], insample[idx_intermittent,:]),
            'intermittent_and_lumpy' : quantile_loss_scaled_mae(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:], insample[idx_intermittent_and_lumpy,:]),
            'non-smooth' : quantile_loss_scaled_mae(actuals[non_smooth,:], quantile_forecasts[non_smooth,:,:], insample[non_smooth,:])
            },
        'rmsse': {
            'all' : rmsse(actuals, mean_forecasts, insample),
            'intermittent' : rmsse(actuals[idx_intermittent,:], mean_forecasts[idx_intermittent,:], insample[idx_intermittent,:]),
            'intermittent_and_lumpy' : rmsse(actuals[idx_intermittent_and_lumpy,:], mean_forecasts[idx_intermittent_and_lumpy,:], insample[idx_intermittent_and_lumpy,:]),
            'non-smooth' : rmsse(actuals[non_smooth,:], mean_forecasts[non_smooth,:], insample[non_smooth,:])
            }
    }
    json.dump(metrics, open(os.path.join(model_folder_path,"metrics.json"), "w"), indent=4)
    logger.log(f"End. Find results in {model_folder_path}")
    logger.off()

    # to debug:
    # --dataset_name RAF --model toto
    # --dataset_name UCI --model toto --num_samples 50
    # --dataset_name carparts --model timesfm
    # --dataset_name carparts --model tirex
