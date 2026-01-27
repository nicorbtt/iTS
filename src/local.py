import subprocess
import argparse
import os
import sys
import numpy as np
from datetime import datetime
import json

from dataloader import load_raw, create_datasets
from visual import Logger
from models import LocalModel
from measures import compute_intermittent_indicators, quantile_loss_, quantile_loss, quantile_loss_scaled_in_sample, rmsse  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="local vs global models for intermittent time series forecasting")
    parser.add_argument('--dataset_name', type=str, choices=['OnlineRetail', 'Auto', 'RAF', 'carparts', 'syph', 'M5', 'crime', 'VN1', 'UCI'], required=True, help='Specify dataset name')
    parser.add_argument('--model', type=str, choices=['ISQ', 'iETS', 'tweedieGP'], required=True, help="Specify model")
    parser.add_argument('--silent', '-s', action='store_true', help='Silent, i.e. no verbose')
    parser.add_argument('--log', '-log', action='store_true', help='Whether to save the log')
    parser_args = parser.parse_args()


    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_folder_name = (
        parser_args.model + "__" +
        parser_args.dataset_name + "__" +
        dt
    )
    model_folder_path = os.path.join(os.getcwd(), "trained_models", model_folder_name)
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

    # create datasets and buil the model
    datasets = create_datasets(data_raw, data_info)
    model = LocalModel(parser_args.model, data_info)

    logger.log("Generate forecasts")
    qlevels = [0.5, 0.8, 0.9, 0.95, 0.99]
    mean_forecasts_list, quantile_forecasts_list = [], []
    for i, y in enumerate(datasets['valid'][:]['target']):
        logger.log("Series " + str(i+1) + " out of " + str(len(datasets['valid'])))
        mean_forecast, quantiles_forecast = model.forecast(y)
        mean_forecasts_list.append(mean_forecast)
        quantile_forecasts_list.append(quantiles_forecast)
    mean_forecasts = np.array(mean_forecasts_list)
    quantile_forecasts = np.array(quantile_forecasts_list)

    actuals = np.array([x["target"][-data_info['h']:] for x in datasets['test']])
    insample = np.array([x["target"][:-data_info['h']] for x in datasets['test']])
    assert quantile_forecasts.shape == (*actuals.shape, len(qlevels))
    assert mean_forecasts.shape == actuals.shape
    np.save(os.path.join(model_folder_path, os.path.join("forecasts", "mean_forecasts.npy")), mean_forecasts)
    for k, q in enumerate(qlevels):
        np.save(os.path.join(model_folder_path, os.path.join("forecasts", "q"+str(q)+".npy")), quantile_forecasts[:,:,k])
    np.save(os.path.join(model_folder_path, os.path.join("forecasts", "actuals.npy")), actuals)
    json.dump({'datetime': dt, 
                'dataset': parser_args.dataset_name, 
                'model': parser_args.model}, open(os.path.join(model_folder_path, "experiment.json"), "w"), indent=4)


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

    
        

        




