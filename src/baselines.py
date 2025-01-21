import os

from dataloader import load_raw, create_datasets
from measures import quantile_loss, compute_intermittent_indicators, rho_risk

import sys
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import json
from datetime import datetime

from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

quantiles = [0.5, 0.8, 0.9, 0.95, 0.99]
quantiles_str = ", ".join([str(q) for q in quantiles])
quantiles_str = f"c({quantiles_str})"

importr('forecast')
importr('smooth')

robjects.r(f"""
    quantiles <- {quantiles_str}
    iETS <- function(z, h, levels=quantiles) {{
        set.seed(0)
        suppressWarnings(model <- smooth::adam(z, model="MNN", occurrence="auto"))
        suppressWarnings(pred <- forecast::forecast(model, h=h, interval='simulated', nsim=50000, level=levels, scenarios=T, side='upper'))
        return(pred)
    }}
""")

def iETS(x):
    return np.array(robjects.r['iETS'](x[0], x[1]).rx2('upper'))

def EmpQ(x):
    return np.tile(np.quantile(x[0], quantiles), (x[1], 1))


#
#   e.g. from terminal:     python baselines.py Auto iETS
#

if __name__ == "__main__":
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    dataset_name = sys.argv[1]

    data_raw, data_info = load_raw(dataset_name=dataset_name, datasets_folder_path=os.path.join(os.getcwd(),"data"))
    datasets = create_datasets(data_raw, data_info)
    adi, cv2 = compute_intermittent_indicators(data_raw, data_info['h']) 
    idx_intermittent = np.logical_and(adi >= 1.32, cv2 < .49)
    idx_intermittent_and_lumpy = adi >= 1.32

    f = None
    assert sys.argv[2] in ["iETS","EmpQ"]
    if sys.argv[2] == "iETS": f = iETS
    if sys.argv[2] == "EmpQ": f = EmpQ


    quantile_forecasts = np.stack(
        process_map(f, 
                    [(robjects.vectors.FloatVector(x), data_info['h']) for x in datasets['valid'][:]['target']], 
                    max_workers=cpu_count()-2, chunksize=1
        )
    )
    actuals = np.stack([x[-data_info['h']:] for x in datasets['test'][:]['target']])
    
    assert actuals.shape[0] == quantile_forecasts.shape[0]
    assert actuals.shape[1] == quantile_forecasts.shape[1]
    assert quantile_forecasts.shape[2] == len(quantiles)
    
    model_folder_path =  os.path.join(os.path.expanduser("~/switchdrive"), "iTS", "trained_models_baselines",sys.argv[2]+"__"+dataset_name+"__"+dt)
    if not os.path.exists(path=model_folder_path):
        os.makedirs(model_folder_path)
        np.save(os.path.join(model_folder_path,"qforecasts.npy"), quantile_forecasts)
        np.save(os.path.join(model_folder_path,"actuals.npy"), actuals)
        np.save(os.path.join(model_folder_path,"q.npy"), quantiles)
        metrics = {'quantile_loss' : {'all' : quantile_loss(actuals, quantile_forecasts, quantiles),
                                    'intermittent' : quantile_loss(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:], quantiles),
                                    'intermittent_and_lumpy' : quantile_loss(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:], quantiles)},
                'rho_risk_nan' : {'all' : rho_risk(actuals, quantile_forecasts, quantiles, zero_denom = np.nan),
                                    'intermittent' : rho_risk(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:], quantiles, zero_denom = np.nan),
                                    'intermittent_and_lumpy' : rho_risk(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:], quantiles, zero_denom = np.nan)},
                'rho_risk_1' : {'all' : rho_risk(actuals, quantile_forecasts, quantiles, zero_denom = 1.),
                                'intermittent' : rho_risk(actuals[idx_intermittent,:], quantile_forecasts[idx_intermittent,:,:], quantiles, zero_denom = 1.),
                                'intermittent_and_lumpy' : rho_risk(actuals[idx_intermittent_and_lumpy,:], quantile_forecasts[idx_intermittent_and_lumpy,:,:], quantiles, zero_denom = 1.)}}
        json.dump(metrics, open(os.path.join(model_folder_path,"metrics.json"), "w"))