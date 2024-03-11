import numpy as np
import matplotlib.pyplot as plt



# Plot learning curves
def learning_curves(history, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    epoch = len(history['train_loss'])
    plt.plot(range(epoch), history['train_loss'], label='train')
    plt.plot(range(epoch), history['val_loss'], label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curves')
    plt.legend()
    plt.show()

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