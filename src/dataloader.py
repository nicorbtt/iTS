import numpy as np
import pandas as pd
from datetime import datetime
from functools import lru_cache, partial
from typing import Optional, Iterable
import torch
from datasets import Dataset, DatasetDict
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from gluonts.transform.sampler import InstanceSampler
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    # SelectFields,
    # SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.time_feature import (
    time_features_from_frequency_str,
    # TimeFeature,
    # get_lags_for_frequency,
)

from transformers import PretrainedConfig



### Datasets Metadata
DATASETS_METADATA = {
    'OnlineRetail'  : {'h' : 28,    'freq' : 'D', 'start' : '2010-12-01'},
    'Auto'          : {'h' : 6,     'freq' : 'M', 'start' : '2010-01-01'},
    'RAF'           : {'h' : 12,    'freq' : 'M', 'start' : '1996-01-01'},
    'carparts'      : {'h' : 6,     'freq' : 'M', 'start' : '1998-01-01'},
    'syph'          : {'h' : 8,     'freq' : 'W', 'start' : '2007-01-01'},
    'M5'            : {'h' : 28,    'freq' : 'D', 'start' : '2011-01-29'},
}

### Import raw data from disk
def load_raw(dataset_name, datasets_folder_path):
    assert dataset_name in ["OnlineRetail", "Auto", 'RAF', 'carparts', 'syph', 'M5']
    data_raw = pd.read_csv(datasets_folder_path + "/" + dataset_name + "/data.csv")
    data_info = {
        'h' : DATASETS_METADATA[dataset_name]['h'],
        'freq' : DATASETS_METADATA[dataset_name]['freq'],
        'start' : DATASETS_METADATA[dataset_name]['start']
    }
    return(data_raw, data_info)

### Create {training, validation, testing} datasets in the gluonts format
def create_datasets(data, data_info, na_rm = True):
    @lru_cache(10_000)
    def convert_to_pandas_period(date, freq):
        return pd.Period(date, freq)
    def transform_start_field(batch, freq):
        batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
        return batch
    # Init data scructures 
    ds_train = {'start':None, 'target':[], 'feat_static_cat':[], 'feat_dynamic_real':[], 'item_id':[]}
    ds_valid = {'start':None, 'target':[], 'feat_static_cat':[], 'feat_dynamic_real':[], 'item_id':[]}
    ds_test  = {'start':None, 'target':[], 'feat_static_cat':[], 'feat_dynamic_real':[], 'item_id':[]}
    h = data_info['h']
    start_dt = datetime.strptime(data_info['start'], '%Y-%m-%d')
    ts_names = [f"S{i}" for i in range(1, data.shape[0]+1)]
    tsIdInc = 0
    # For each ts in the raw data append target, feat_static_cat (i.e., time series id) and start
    for i in range(data.shape[0]):
        if na_rm and np.sum(np.isnan(data.values[i,:])) > 0:
            continue
        ds_train['target'].append(data.values[i,:-h*2])
        ds_train['feat_static_cat'].append([tsIdInc]); ds_train['item_id'].append(ts_names[i]); ds_train['feat_dynamic_real'].append(None)
        ds_valid['target'].append(data.values[i,:-h])
        ds_valid['feat_static_cat'].append([tsIdInc]); ds_valid['item_id'].append(ts_names[i]); ds_valid['feat_dynamic_real'].append(None)
        ds_test['target'].append(data.values[i,:])
        ds_test['feat_static_cat'].append([tsIdInc]); ds_test['item_id'].append(ts_names[i]); ds_test['feat_dynamic_real'].append(None)
        tsIdInc = tsIdInc + 1
    ds_train['start'] = [start_dt]*len(ds_train['target'])
    ds_valid['start'] = [start_dt]*len(ds_valid['target'])
    ds_test['start']  = [start_dt]*len(ds_test['target'])
    # Build Dataset objects
    dataset = DatasetDict({
        'train': Dataset.from_dict(ds_train),
        'valid': Dataset.from_dict(ds_valid),
        'test': Dataset.from_dict(ds_test)
    })
    # Transform data (string) to datetime.datetime
    dataset['train'].set_transform(partial(transform_start_field, freq=data_info['freq']))
    dataset['valid'].set_transform(partial(transform_start_field, freq=data_info['freq']))
    dataset['test'].set_transform(partial(transform_start_field, freq=data_info['freq']))
    return(dataset)

### Transformations Chain
def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config['num_static_real_features'] == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config['num_dynamic_real_features'] == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config['num_static_categorical_features'] == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    if freq =="M":
        freq="ME"

    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config['num_static_categorical_features'] > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config['num_static_real_features'] > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # n.b. we expect an extra dim for the multivariate case!
                expected_ndim=1,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config["prediction_length"],
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config["prediction_length"],
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config['num_dynamic_real_features'] > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

### Instance Splitters
def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config["prediction_length"]
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config["prediction_length"]),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config['context_length'] + max(config['lags_sequence']),
        future_length=config["prediction_length"],
        time_series_fields=["time_features", "observed_mask"],
    )

### DataLoaders
def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config['num_static_categorical_features'] > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config['num_static_real_features'] > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )
def create_backtest_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ] + ["future_values", "future_observed_mask"]
    if config['num_static_categorical_features'] > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config['num_static_real_features'] > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data)

    # we create a Validation Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "validation")

    # we apply the transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )
def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config['num_static_categorical_features'] > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config['num_static_real_features'] > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # We create a test Instance splitter to sample the very last
    # context window from the dataset provided.
    instance_sampler = create_instance_splitter(config, "test")

    # We apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )
def create_dataloaders(config, datasets, data_info, batch_size=128, num_batches_per_epoch=100):

    # hack ;)
    if not isinstance(config, PretrainedConfig):
        config = {
            {
                'num_feat_static_cat' : 'num_static_categorical_features',
                'num_feat_static_real' : 'num_static_real_features',
                'num_feat_dynamic_real' : 'num_dynamic_real_features',
                'lags_seq' : 'lags_sequence'
            }.get(k, k): v for k, v in config.items()}
    else:
        config = config.__dict__
    
    train_dataloader = create_train_dataloader(config=config, 
                                               freq=data_info['freq'], 
                                               data=datasets['train'], 
                                               batch_size=batch_size, 
                                               num_batches_per_epoch=100)
    valid_dataloader = create_backtest_dataloader(config=config, 
                                                  freq=data_info['freq'], 
                                                  data=datasets['valid'],
                                                  batch_size=batch_size)
    test_dataloader = create_test_dataloader(config=config, 
                                             freq=data_info['freq'], 
                                             data=datasets['test'], 
                                             batch_size=batch_size)
    return (train_dataloader, valid_dataloader, test_dataloader)