import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_fitness_curves(fitness_data, title, xlim=None, ylim=None):
    """
    Parameters
    ----------
    fitness_data : list of dataframes that contains fitness curves for each run
        each row is a separate iteration, each column a separate run
        one dataframe for each key HP value

    title : string
        Title for the chart.
    """

    fig, ax = plt.subplots()
    plt.grid()
    plt.title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness score")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # compute means and standard deviations across all runs
    fitness_means = np.mean(fitness_data, axis=1)
    fitness_stdevs = np.std(fitness_data, axis=1)

    # plot scores
    ax.fill_between(
        fitness_data.index,
        fitness_means - fitness_stdevs,
        fitness_means + fitness_stdevs,
        alpha=0.1,
        color='r',
    )
    ax.plot(fitness_means, '-', color='r', label="Mean fitness")
    ax.legend(loc='best')
    plt.tight_layout()

    return plt


def plot_time_complexity_curves(df, title):
    pass
