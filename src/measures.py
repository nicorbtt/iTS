import numpy as np
import pandas as pd
from scipy.stats import ecdf



def compute_adi_cv2(data):
    adi = 1 / np.mean(data != 0, axis=1)
    nan_data = np.where(data != 0, data, np.nan)
    cv2 = (np.nanstd(nan_data, axis = 1) / np.nanmean(nan_data, axis = 1)) ** 2
    return adi, cv2

def compute_intermittent_indicators_v2(data):
    # TODO add NaN ts filter
    if (isinstance(data, pd.DataFrame)):
        data = data.values
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

def compute_intermittent_indicators(data, h, na_rm = True, zero_rm=True):
    if (isinstance(data, pd.DataFrame)):
        data = data.values
    assert isinstance(data, np.ndarray) and data.ndim == 2

    if na_rm:
        data = data[~np.any(np.isnan(data), axis=1),:]

    if zero_rm:
        data = data[~np.all(data[:,:-h*2] == 0, axis=1)]
    
    adi, cv2 = np.empty(len(data)), np.empty(len(data))
    idx = np.arange(data.shape[1])
    for i, ts in enumerate(data):
        demand_idx = np.concatenate(([0], idx[ts > 0]+1))
        demand = ts[ts > 0]
        if len(demand) == 0:
            adi[i], cv2[i] = np.nan, np.nan
        else: adi[i], cv2[i] = np.mean(demand_idx[1:]-demand_idx[:-1]), (np.std(demand)/np.mean(demand))**2
    return adi, cv2

def label_intermittent(adi, cv2, f):
    assert f in ["intermittent", "lumpy"]

    if (f == "intermittent"):
        return np.logical_and(adi >= 1.32, cv2 < .49)
    if (f == "lumpy"):
        return np.logical_and(adi >= 1.32, cv2 >= .49)
    
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
               if np.any(np.isnan(v)):
                   v = v[~np.isnan(v)]
               df_dict[k + ' (' + s + ')'] = [np.min(v), np.quantile(v, .25), np.median(v), np.quantile(v, .75), np.max(v)]
        df = pd.DataFrame(df_dict, index=['min', '25%ile', 'median', '75%ile', 'max'])
        df.columns = pd.MultiIndex.from_product([('demand intervals', 'demand sizes', 'demand per period'), ('mean', 'std')])
        return df
    else:
        return stats

### Metrics
# Discrete RPS
def drps(actual, samples, lower=None, upper=None):
    assert len(actual) == len(samples)
    if lower is None:
        lower = int(np.floor(np.nanmin([np.min(samples), np.min(actual)])))
    if upper is None:
        upper = int(np.ceil(np.nanmax([np.max(samples), np.max(actual)])))
    return np.array([np.sum((ecdf(samples[i]).cdf.evaluate(np.arange(lower, upper+1)) - 
                            (np.arange(lower, upper+1) >= actual[i]).astype(float))**2)
                             for i in range(len(actual))])

# Pinball loss (e.g., l=0.95)
def pinball(actual, pred, l):
    return np.abs(actual-pred)*np.where(actual >= pred, l, 1-l)

# Quantile loss
def quantile_loss_(target: np.ndarray, forecast: np.ndarray, q: float, avg=True) -> float:
    if avg:
        return 2 * np.mean(np.abs((forecast - target) * ((target <= forecast) - q)))
    else:
        return 2 * np.abs((forecast - target) * ((target <= forecast) - q))

def quantile_loss(target: np.ndarray, forecast: np.ndarray, quantiles = [0.5, 0.8, 0.9, 0.95, 0.99], avg=True):
    res = {}
    for q in range(len(quantiles)):
        res['QL'+str(int(quantiles[q]*100))] = quantile_loss_(target, np.round(forecast[:,:,q]), quantiles[q], avg)
    return(res)

def quantile_loss_sample(target: np.ndarray, forecast: np.ndarray, quantiles = [0.5, 0.8, 0.9, 0.95, 0.99], avg=True):
    forecast = np.swapaxes(forecast, 1, 2)
    tmp = np.empty(shape=(forecast.shape[0], forecast.shape[1], len(quantiles)))
    for i in range(tmp.shape[0]):
        for j in range(len(quantiles)):
            tmp[i,:,j] = np.round(np.quantile(forecast[i], axis=1, q=quantiles[j]))
    return(quantile_loss(target, tmp, quantiles, avg))

def brier_score(target: np.ndarray, forecast: np.ndarray, avg=True) -> float: #forecast = prob(0)
    if avg:
        return float(np.mean((forecast - (target==0))**2))
    else:
        return (forecast - (target==0))**2


def rho_risk_(target: np.ndarray, forecast: np.ndarray, q: float, zero_denom: float) -> float:
    quantile_loss = 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)), axis=1)
    actual_sum = np.sum(target, axis= 1, dtype=np.float32)
    actual_sum[actual_sum == 0] = zero_denom
    return np.nanmean(quantile_loss/actual_sum)

def rho_risk(target: np.ndarray, forecast: np.ndarray, quantiles = [0.25, 0.5, 0.8, 0.9, 0.95, 0.99], zero_denom= np.nan):
    res = {}
    for q in range(len(quantiles)):
        res['q'+str(quantiles[q])] = rho_risk_(target, np.round(forecast[:,:,q]), q=quantiles[q], zero_denom = zero_denom)
    return(res)

def rho_risk_sample(target: np.ndarray, forecast: np.ndarray, quantiles = [0.25, 0.5, 0.8, 0.9, 0.95, 0.99], zero_denom = np.nan):
    tmp = np.transpose(np.round(np.quantile(forecast, quantiles, axis=1)), (1,2,0))
    return (rho_risk(target, tmp, quantiles, zero_denom))


    

# Mase
def mase(actual, pred, in_sample_data):
    return np.mean(np.abs(actual - pred), axis=1)/np.mean(np.abs(in_sample_data[:,1:] - in_sample_data[:,:-1]), axis=1)

# Proportion in stock
def pis(actual, pred):
    return -np.sum(np.cumsum(actual - pred, axis=1), axis=1)