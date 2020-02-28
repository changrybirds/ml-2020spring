import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def means_stdevs(fitness_df):
    fitness_means = fitness_df.mean(axis=1)
    fitness_stdevs = fitness_df.std(axis=1)

    return fitness_means, fitness_stdevs


def plot_iterative_lc(df, title, max_iter_range, ylim=None):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("max_iterations")
    ax1.set_ylabel("Score")

    train_scores = df['train']
    train_scores_std = np.std(df['train'])
    cv_scores = df['cv']
    cv_scores_std = np.std(df['cv'])

    # plot scores on left axis
    # ax1.fill_between(max_iter_range, train_scores - train_scores_std,
    #                  train_scores + train_scores_std, alpha=0.1, color="r")
    # ax1.fill_between(max_iter_range, cv_scores - cv_scores_std,
    #                  cv_scores + cv_scores_std, alpha=0.1, color="g")
    line1 = ax1.plot(max_iter_range, train_scores, 'o-', color="r",
             label="Training score")
    line2 = ax1.plot(max_iter_range, cv_scores, 'o-', color="g",
             label="CV score")

    # plot times on the right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    line3 = ax2.plot(df['train_time'], 'o-', color='b', label="Training Time")
    line4 = ax2.plot(df['cv_time'], 'o-', color='y', label="CV Time")

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    return plt


def plot_fitness_curves(fitness_data, title, hp_values=None, hp_name=None, xlim=None, ylim=None):
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

    if hp_values is not None:
        lines = []
        for fitness_df, hp_val in zip(fitness_data, hp_values):

            # compute means and standard deviations across all runs
            fitness_means, fitness_stdevs = means_stdevs(fitness_df)

            # shade in 1 stdev bounds
            ax.fill_between(
                fitness_df.index,
                fitness_means - fitness_stdevs,
                fitness_means + fitness_stdevs,
                alpha=0.1,
            )
            # plot means
            line = ax.plot(fitness_means, label=str(hp_val))
            lines.append(line[0])

        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, title=hp_name, loc='best')

    else:
        # compute means and standard deviations across all runs
        fitness_means, fitness_stdevs = means_stdevs(fitness_data)

        # shade in 1 stdev bounds
        ax.fill_between(
            fitness_data.index,
            fitness_means - fitness_stdevs,
            fitness_means + fitness_stdevs,
            alpha=0.1,
        )
        # plot means
        line = ax.plot(fitness_means)

    plt.tight_layout()

    return plt