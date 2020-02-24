import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def means_stdevs(fitness_df):
    fitness_means = fitness_df.mean(axis=1)
    fitness_stdevs = fitness_df.std(axis=1)

    return fitness_means, fitness_stdevs


def plot_fitness_curves(fitness_dfs, hp_values, hp_name, title, xlim=None, ylim=None):
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
    for fitness_df, hp_val in zip(fitness_dfs, hp_values):
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
    plt.tight_layout()

    return plt


def plot_algo_comparisons(title, rhc_df=None, sa_df=None, ga_df=None, mimic_df=None, xlim=None, ylim=None):
    fig, ax = plt.subplots()
    plt.grid()
    plt.title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness score")

    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)

    lines = []  # generate list of lines plotted

    # for each algorithm used:
    #   1. get fitness means and stdevs
    #   2. shade in 1 standard deviation bounds
    #   3. plot means of all runs
    #   4. concatenate lines to use for pulling labels for legend

    if rhc_df is not None:
        rhc_means, rhc_stdevs = means_stdevs(rhc_df)
        ax.fill_between(rhc_df.index, rhc_means - rhc_stdevs, rhc_means + rhc_stdevs, color='b', alpha=0.1)
        line_rhc = ax.plot(rhc_means, color='b', label="RHC")
        lines = lines + line_rhc

    if sa_df is not None:
        sa_means, sa_stdevs = means_stdevs(sa_df)
        ax.fill_between(sa_df.index, sa_means - sa_stdevs, sa_means + sa_stdevs, color='r', alpha=0.1)
        line_sa = ax.plot(sa_means, color='r', label="SA")
        lines = lines + line_sa

    if ga_df is not None:
        ga_means, ga_stdevs = means_stdevs(ga_df)
        ax.fill_between(ga_df.index, ga_means - ga_stdevs, ga_means + ga_stdevs, color='g', alpha=0.1)
        line_ga = ax.plot(ga_means, color='g', label="GA")
        lines = lines + line_ga

    if mimic_df is not None:
        mimic_means, mimic_stdevs = means_stdevs(mimic_df)
        ax.fill_between(mimic_df.index, mimic_means - mimic_stdevs, mimic_means + mimic_stdevs, color='y', alpha=0.1)
        line_mimic = ax.plot(ga_means, color='y', label="MIMIC")
        lines = lines + line_mimic

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, title="Algorithm", loc='best')
    plt.tight_layout()

    return plt


def compare_algos(problem_name, rhc_dfs=None, sa_dfs=None, ga_dfs=None, mimic_dfs=None):
    # for each algorithm used:
    #   1. collapse dfs to a single column w/ means across all runs
    #   2. find the dataframe that performs the best for each algorithm
    #   3. plot comparison of algorithm performance

    if rhc_dfs is not None:
        mean_rhc_dfs = []
        for df in rhc_dfs: mean_rhc_dfs.append(means_stdevs(df)[0])
        best_rhc_fitness_vals = np.array([mean_df.max() for mean_df in mean_rhc_dfs])
        best_rhc_df = rhc_dfs[np.argmax(best_rhc_fitness_vals)]
    else:
        best_rhc_df = None

    if sa_dfs is not None:
        mean_sa_dfs = []
        for df in sa_dfs: mean_sa_dfs.append(means_stdevs(df)[0])
        best_sa_fitness_vals = np.array([mean_df.max() for mean_df in mean_sa_dfs])
        best_sa_df = sa_dfs[np.argmax(best_sa_fitness_vals)]
    else:
        best_sa_df = None

    if ga_dfs is not None:
        mean_ga_dfs = []
        for df in ga_dfs: mean_ga_dfs.append(means_stdevs(df)[0])
        best_ga_fitness_vals = np.array([mean_df.max() for mean_df in mean_ga_dfs])
        best_ga_df = ga_dfs[np.argmax(best_ga_fitness_vals)]
    else:
        best_ga_df = None

    if mimic_dfs is not None:
        mean_mimic_dfs = []
        for df in mimic_dfs: mean_mimic_dfs.append(means_stdevs(df)[0])
        best_mimic_fitness_vals = np.array([mean_df.max() for mean_df in mean_mimic_dfs])
        best_mimic_df = mimic_dfs[np.argmax(best_mimic_fitness_vals)]
    else:
        best_mimic_df = None

    plot_algo_comparisons(
        title="Algorithm comparison: fitness vs iterations",
        rhc_df=best_rhc_df,
        sa_df=best_sa_df,
        ga_df=best_ga_df,
        mimic_df=best_mimic_df,
    )

    # save plot as PNG
    plot_filename = 'graphs/' + problem_name + '_algo_comparison.png'
    plt.savefig(plot_filename)
    plt.clf()


if __name__ == "__main__":
    pass
