import datasets
import pandas as pd
import numpy as np
from metrics_scores import ts_classification

def load_DatasetDict(name, pred_length = None, idx=None):

    assert name in ["OnlineRetail", "Auto", 'RAF', 'carparts', 'syph', 'M5']
    path =   '../data/' + name + '/'

    if pred_length is None and idx is None:
        return datasets.load_dataset(path=path, data_files={'train':'train.json', 'test':'test.json'})
    
    else:
        train, test = pd.read_json(path + 'test.json', orient='records'), pd.read_json(path + 'test.json', orient='records')
        if idx is not None: 
            test, train = test.iloc[idx,:], train.iloc[idx,:]
        if pred_length is not None:
            train.target = [ts[:-pred_length] for ts in test.target]
        return datasets.DatasetDict({'train':datasets.Dataset.from_pandas(train),
                                     'test':datasets.Dataset.from_pandas(test)})
    

def dataset_stats(name):

    assert name in ["OnlineRetail", "Auto", 'RAF', 'carparts', 'syph', 'M5']
    path =   '../data/' + name + '/'

    default_pred_length = {'OnlineRetail': 28,
                          'Auto' : 6, 
                          'RAF' : 12, 
                          'carparts': 6, 
                          'syph' : 8, 
                          'M5' : 28}
    
    freq = {'OnlineRetail': '1D',
            'Auto' : '1M',
            'RAF' : '1M',
            'carparts': '1M',
            'syph' : '1W', 
            'M5' : '1D'}
    
    start = {'OnlineRetail': '2010-12-01',
             'Auto' : '2010-01-01',
             'RAF' : '1996-01-01',
             'carparts': '1998-01-01',
             'syph' : '2007-01-01', 
             'M5' : '2011-01-29'}
    
    adi, cv2 = ts_classification(np.array(pd.read_csv(path + 'data.csv').values))

    return {'default_pred_length' : default_pred_length[name],
            'freq' : freq[name],
            'start' : start[name],
            'intermittent idx' : np.logical_and(adi >= 1.32, cv2 < .49),
            'lumpy idx': np.logical_and(adi >= 1.32, cv2 >= .49)}
    
