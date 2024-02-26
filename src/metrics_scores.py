import numpy as np
import pandas as pd
from scipy.stats import ecdf

def compute_drps(actual, samples, lower =None, upper = None):
    assert len(actual) == len(samples)
    if lower is None:
        lower = int(np.floor(np.nanmin([np.min(samples), np.min(actual)])))
    if upper is None:
        upper = int(np.ceil(np.nanmax([np.max(samples), np.max(actual)])))
    return np.array([np.sum((ecdf(samples[i]).cdf.evaluate(np.arange(lower, upper+1)) - 
                            (np.arange(lower, upper+1) >= actual[i]).astype(float))**2)
                             for i in range(len(actual))])

def compute_pinball(actual, pred, l):
    return np.abs(actual-pred)*np.where(actual >= pred, l, 1-l)

def compute_mase(actual, pred, in_sample_data):
    return np.mean(np.abs(actual - pred), axis=1)/np.mean(np.abs(in_sample_data[:,1:] - in_sample_data[:,:-1]), axis=1)

def compute_pis(actual, pred):
    return -np.sum(np.cumsum(actual - pred, axis=1), axis=1)

def compute_adi_cv2(data):
    adi = 1/np.mean(data != 0, axis=1)
    nan_data = np.where(data != 0, data, np.nan)
    cv2 = (np.nanstd(nan_data, axis = 1)/np.nanmean(nan_data, axis = 1))**2
    return adi, cv2

def ts_classification(data):

    assert isinstance(data, np.ndarray) and data.ndim == 2

    zero_ts = np.all(data == 0, axis=1)

    if np.any(zero_ts):
        L = data.shape[0]
        adi, cv2 = np.empty(L), np.empty(L)
        adi[zero_ts], cv2[zero_ts]  = np.nan, np.nan
        adi[~zero_ts], cv2[~zero_ts] = compute_adi_cv2(data[~zero_ts,:])
    else:
        adi, cv2 = compute_adi_cv2(data)

    return adi, cv2

def ts_classification_alt(data):

    assert isinstance(data, np.ndarray) and data.ndim == 2
    
    adi, cv2 = np.empty(len(data)), np.empty(len(data))

    idx = np.arange(data.shape[1])

    for i, ts in enumerate(data):

        demand_idx = np.concatenate(([0], idx[ts > 0]+1))
        demand = ts[ts > 0]

        if len(demand) == 0:
            adi[i], cv2[i] = np.nan, np.nan

        adi[i], cv2[i] = np.mean(demand_idx[1:]-demand_idx[:-1]), (np.std(demand)/np.mean(demand))**2

    return adi, cv2
    

def dataset_stats(data, to_pandas = True):

    assert isinstance(data, np.ndarray) and data.ndim == 2
    

    idx = np.arange(data.shape[1])

    demand_intervals_mean, demand_intervals_std = np.empty(len(data)), np.empty(len(data))
    demand_sizes_mean, demand_sizes_std = np.empty(len(data)), np.empty(len(data))
    demand_per_period_mean, demand_per_period_std = np.empty(len(data)), np.empty(len(data))

    for i, ts in enumerate(data):

        demand_idx = np.concatenate(([0], idx[ts > 0]+1))
        demand_intervals = demand_idx[1:] - demand_idx[:-1]
        demand_sizes = ts[ts > 0]
        demand_per_period = demand_sizes/demand_intervals

        assert len(demand_intervals) == len(demand_sizes)

        demand_intervals_mean[i], demand_intervals_std[i] = np.mean(demand_intervals), np.std(demand_intervals)
        demand_sizes_mean[i], demand_sizes_std[i] = np.mean(demand_sizes), np.std(demand_sizes)
        demand_per_period_mean[i], demand_per_period_std[i] = np.mean(demand_per_period), np.mean(demand_per_period)

    stats = {'demand intervals' : {'mean' : demand_intervals_mean,
                                   'std' : demand_intervals_std},
             'demand sizes' : {'mean' : demand_sizes_mean,
                               'std' : demand_sizes_std},
             'demand per period' : {'mean' : demand_per_period_mean,
                                    'std' : demand_per_period_std}}

    if to_pandas:
        
        df_dict = {}
        
        for k in stats.keys():
            for s in ['mean', 'std']:
               v = stats[k][s]
               df_dict[k + ' (' + s + ')'] = [np.min(v), np.quantile(v, .25), np.median(v), np.quantile(v, .75), np.max(v)]

        return pd.DataFrame(df_dict, index=['min', '25%ile', 'median', '75%ile', 'max'])
    
    else:
        return stats



