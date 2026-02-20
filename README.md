# Intermittent time series forecasting: local vs global models
Stefano Damato, Nicoló Rubattu, Dario Azzimonti, Giorgio Corani

## Overview

This is the official repository fo the paper "Intermittent time series forecasting: local vs global models", authored by S. Damato, N. Rubattu, D. Azzimonti and G. Corani, currently under review at the Data Mining and Knowledge Discovery track of ECMLPKDD2026.

The repository is organised as follows:


```text
.
├── README.md
├── data
│   ├── datasets.R
│   └── datasets.ipynb
├── Dockerfile
├── environment.yml
├── main.py
├── packages
│   ├── gluon-ts/
│   ├── transformers/
│   └── tweediegp/
├── script.sh
├── src
│   ├── dataloader.py
│   ├── global.py
│   ├── local.py
│   ├── measures.py
│   ├── models.py
│   ├── results.ipynb (#TODO)
│   └── visual.py
└── trained_models/
```

In particular, the following files are of interest:
- `README.md`: this file.
- `environment.yml`: to create the conda environment to run the code.
- `packages/` folder: contains local versions of three open-source packages, to which we implemented some modifications.
- `data/` folder: contains the data sets to be used in the experiments, as well as scripts to download and preprocess them.
- `src/` folder: contains the code to reproduce the experiments in the paper.
- `trained_models/` folder: where experiments results are saved. For now, it is left empty due to size constraints.
- `main.py`: script to launch the experiments.
- `script.sh`: an example script to run multiple all the experiments of the project.
- `Dockerfile`: to build a Docker image with the environment and code to run the experiments.


## Reproducibility

### Environment

Crate a conda environment with the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

This will install all the required packages to run the code in the repository, including the modified versions of `gluonts` and `transformers` which we adapted to perform our experiments, and a package version of `tweediegp`. To actually use it, run:
```bash
conda activate ilocglob
```
before any other action below.

### Data

Use the material in the `data` folder to download and preprocess the datasets used in the experiments.
Run first `datasets.R` in R to download the raw files of some datasets, and then `datasets.ipynb` to preprocess all datasets into a common format.

### Source code

The `src/` folder contains the material to reproduce the experiments and results in the paper.

It contains what follows:

- `dataloader.py`: code to load the data sets and sample from them.
- `measures.py`: contains the metrics used to evaluate the performance of the methods.
- `models.py`: contains the code to build all the models under a common interface.
- `visual.py`: contains some helper functions for checking and visualizing training loops.
- `local.py`: run the experiments with local models.
- `global.py`: run the experiments with global models.
- `results.ipynb`: notebook to reproduce the tables and figures in the paper from the saved results of the experiments (#TODO).

### Launching experiments

To run the experiments, use the script `main.py`, which is a wrapper to launch calling either `local.py` or `global.py`, with the critical arguments.
The common specifications are, geiven the appropriate values in brackets:

- `--dataset_name {DATASETNAME}`: the name of the dataset to use, among those of the paper.
- `--model {MODELNAME}`: the model to be used.

For local models, it will take the form:
```bash
python main.py -log --dataset_name {DATASETNAME} --model {MODELNAME}
```
For global models, it will take the form:
```bash
python main.py -log --dataset_name {DATASETNAME} --model {MODELNAME} --distribution_head {DISTRIBUTIONHEAD} --seed {SEED}
```

where `{SEED}` is a numerical seed for reproducibility, and `{DISTRIBUTIONHEAD}` is the distribution to be coupled with neural networks. The arguments `-log` and `-s`, if used, will enable logging the experiement of text file, and make the training silent respectively.

The same experiments can be launched directly from the scripts `src/local.py` and `src/global.py`, where additional arguments can be specified, such as the scaling method, the number of training epochs, or model hyperparameters. However, `main.py` can be used to reproduce the experiments in the paper with default specifications. 

A comprehensive list of all command-lines to be used is contained in `script.sh`. All the experiments are particularly demanding: running them on the CPU of a local machine is feasible, but may take weeks. We thus suggest to use an HPC infrastructure with GPUs to make the experiments faster and parallelise them; using the `Dockerfile` we provide, a ready-to-use environment can be obtained building a Docker image.


### Trained models


As a result of each experiment, a folder will be created in `trained_models/` containing the forecasts, metrics, metadata on the hyperparameters of the experiments, and information about the training process. Its name will summarise the configuration of the experiment, being similar to `{MODELNNAME}__{DATASETNAME}`, followed by additional information regarding the experiment and the timestamp of its start.

The `trained_models/` folder is left empty; but experiments run with scripts above will be saved in it. Each subfolder, representing an experiment, will contain:

- `actuals.npy`: an array with the obesrved test values.
- `mean_forecasts.npy`: an array of mean forecasts.
- `q_.npy`: an array of quantile forecasts (for local and multi-output models); `_` is replaced by the quantile level, e.g. 0.95.
- `model_params.json`: model and experiment hyperparameters, for global models.
- `model_state.model`: a state dictionary for the parameters of neural networks (not included for local models).
- `experiment.json`: metadata on the experiment for full reproducibility.
- `metrics.json`: aggregated scores of the performance of the model on different metrics, evaluated on different subsets of the data.


## Contacts

To acknowledge our work, please cite the following preprint:

```
@misc{damato2026intermittent,
      title={Intermittent time series forecasting: local vs global models}, 
      author={Stefano Damato and Nicolò Rubattu and Dario Azzimonti and Giorgio Corani},
      year={2026},
      eprint={2601.14031},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2601.14031}, 
}
```

For any questions, contact Stefano Damato (`stefano.damato@supsi.ch`).
For bug and issues reporting, use the GitHub issue tracker of the repository.
