import os
import numpy as np
import matplotlib.pyplot as plt



# Plot learning curves
def learning_curves(history, path, likelihood, scaling ,figsize=(10, 6)):
    plt.figure(figsize=figsize)
    epoch = len(history['train_loss'])
    plt.plot(range(epoch), history['train_loss'], label='train')
    plt.plot(range(epoch), history['val_loss'], label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curves' + ' [likelihood: ' + likelihood + (', scaling:' +  scaling if scaling is not None else '') + ']')
    plt.legend()
    plt.tight_layout() 
    plt.savefig(os.path.join(path, "learning_curves.png"))
    plt.close()

# Plot forecast
def forecast_plot(ts_index, forecasts, datasets, data_info, targetName, alphas=[0.05], legend_loc="upper left", figsize=(8,4)):
    _, ax = plt.subplots(figsize=figsize)
    index = range(len(datasets['test'][ts_index][targetName]))
    ax.plot(index[-3*data_info['h']:], datasets['test'][ts_index]["target"][-3*data_info['h']:], label="actual", color='black')
    ax.plot(index[-data_info['h']:], np.round(np.median(forecasts[ts_index], axis=0), 8), label="predicted", color="blue")
    for a in alphas:
        ax.fill_between(
            index[-data_info['h']:],
            np.quantile(forecasts[ts_index], 1-(a/2), axis=0),
            np.quantile(forecasts[ts_index], a/2, axis=0),
            alpha=0.1, 
            interpolate=True,
            # label=f"{(1-a)*100}%",
            color="blue"
        )
    ax.legend(loc=legend_loc)
    plt.show()

import logging
from typing import IO
import sys

class Logger():

    def __init__(self, disable=False, stdout: IO[str] = sys.stdout) -> None:
        self.disable=disable
        logging.basicConfig(stream=stdout, format='%(asctime)s - %(message)s', 
                            level=logging.INFO, 
                            datefmt='%d-%b-%y %H:%M:%S')

    def log(self, s):
        if not self.disable: logging.info(s)

    def log_epoch(self, epoch, history):
        if not self.disable: 
            logging.info(f"Epoch {epoch+1} \t Train Loss: {history['train_loss'][-1]:.3f} \t Val Loss: {history['val_loss'][-1]:.3f}")
    
    def log_earlystop_newbest(self, best_val_loss):
        if not self.disable: 
            logging.info(f"Early stopping, new validation best: {best_val_loss:.3f}, keep training!")

    def log_earlystop_stop(self, epoch, best_val_loss):
        if not self.disable: 
            logging.info(f"Early stopping after {epoch+1} epochs. Validation best: {best_val_loss:.3f}")

    def off(self):
        sys.stdout = sys.__stdout__
