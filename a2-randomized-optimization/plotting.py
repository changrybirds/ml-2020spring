import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_fitness_curves(fitness_dfs, hp_vals, hp_name, title, xlim=None, ylim=None):
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

    lines = []
    fitness_means_dfs = []
    for fitness_df, hp_val in zip(fitness_dfs, hp_vals):
        # compute means and standard deviations across all runs
        fitness_means = np.mean(fitness_df, axis=1)
        fitness_stdevs = np.std(fitness_df, axis=1)

        # plot scores
        ax.fill_between(
            fitness_df.index,
            fitness_means - fitness_stdevs,
            fitness_means + fitness_stdevs,
            alpha=0.1,
        )
        line = ax.plot(fitness_means, label=str(hp_val))
        lines.append(line[0])
        fitness_means_dfs.append(fitness_means)

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, title=hp_name, loc='best')
    plt.tight_layout()

    return fitness_means_dfs


def plot_algo_comparisons(rhc_df, sa_df, ga_df, mimic_df, title):
    pass
