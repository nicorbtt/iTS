# iTS

This is the official reposirtory fo the paper `Intermittent time series forecasting: local vs global models`, currently under review at the Data Mining and Knowledge Discovery track of ECMLPKDD2026.
The paper is autored by Stefano Damato, Nicoló Rubattu, Dario Azzimonti and Giorgio Corani.

The repository is organised as follows:

- `environment.yml`: contains packages and version specifications to reproduce the experiments.
- `packages/` folder: the local version of the open source packages `gluonts` and `transformers`, which we modified.
- `data/`: contains the datasets used in the experiments as well as scripts to download and preprocess them.
- `src/`: contains the code to reproduce the experiments and results in the paper.
- `trained_models/`: can contain the trained models include in the experiments to avoid retraining them. Left empty for now due to size constraints.

We now detail each of the previous points.

## Environment

Crate a conda environment with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate iTS
```

This will install all the required packages to run the code in the repository, inluding the modified versions of `gluonts` and `transformers` which we adapted to run our experiments.

## Data

Use the material in the `data` folder to download and preprocess the datasets used in the experiments.
Run first `datasets.R` in R to download the raw files of some datasets, and then `datasets.ipynb` to preprocess all datasets into a common format.

## Source code

The `src/` folder contains the material to reproduce the experiments and results in the paper.

It contains what follows:

- `tweediegp/` folder: copies the official implementation of TweedieGP.
- `dataloader.py`: code to load the datasets.
- `measures.py`: contains the metrics used to evaluate the performance of the methods.
- `models.py`: contains the code to build the deep neural network architectures under a common interface.
- `visual.py`: contains some helper functions for checking and visualizing training loops.
- `baselines.py`: script to run the local baseline methods: in-sample quantiles, iETS and TweedieGP.
- `feedforward.py`: script to run the experiments with feed-forward neural networks.
- `dlinear.py`: script to run the experiments with D-Linear.
- `main.py`: script to run the experiments with deep neural networks.


### `feedforward.py` and `dlinear.py`

These script laung the exepriments with the two simples architectures we use. To launch an experiment, run for instance

```bash
python FILENAME.py log --dataset_name DATASETNAME --distribution_head DISTRIBUTIONHEAD --seed SEED --scaling mean-demand --lag LAG
```

where `FILENAME.py` is either `feedforward.py` or `dlinear.py`, `DATASETNAME` is one of the datasets used in the paper, `DISTRIBUTIONHEAD` is one of `tweedie`, `negbin` or `hsnb`, `SEED` is an integer seed for reproducibility, and `LAG` is the context length to use.

As a result, a folder will be created in `trained_models/` containing the forecasts and metrics for the given configuration, metadata concerning the hyperparameters of the experiments, and information about the training process. Its name will summarize the configuration of the experiment, being similar to `{MODELNAME}_l{LAG}__{DATASETNAME}__{DISTRIBUTIONHEAD}__{SCALING}`, folowed by the datetime of the experiment start.

### `main.py` 

This script launches the experiments with the deep neural networks we use. To launch an experiment, run for instance

```bash
python main.py -log --dataset_name DATASETNAME --model MODELNAME --distribution_head DISTRIBUTIONHEAD --seed SEED --scaling mean-demand
```

where `DATASETNAME`, `DISTRIBUTIONHEAD` and `SEED` are as before, and `MODELNAME` is one of `deepAR` `transformer`, `informer`, and `autoformer`.

As above, a folder will be created in `trained_models/`, whose name will summarize the configuration of the experiment. It will have the structure of `{MODELNAME}__{DATASETNAME}__{DISTRIBUTIONHEAD}__{SCALING}`, followed by the datetime of the experiment start, as above.

All the experiments in the paper, but in particular those launched by this script, are particulary demanding: running the on the CPU of a local machine is feasible, but may take weeks. We thus suggest to use a cluster with GPUs to make the experiments faster and parallelise them.

## Trained models

The `trained_models/` folder is left empty; but experiments run with scripts above will be saved here. Each subfolder, representing an experiment, will contain:

- `actuals.npy`: an array with the obesrved test values.
- `forecasts.npy`: an array of sample paths (in case the model uses autoregressive sampling).
- `q_.npy`: an array of quantile forecasts (for local and multi-output models); `_` is replaced by the quantile level, e.g. 0.95.
- `model_params.json`: model and experiment hyperparameters.
- `experiment.json`: metadata on the experiment for full reproducibility.
- `model_state.model`: a state dictionary for the parameters of neural networks (not included for local models).
- `metrics.json`: aggregated scores of the performance of the model on different metrics, evaluated on different subsets of the data.

