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


def plot_algo_comparisons(title, rhc_df, sa_df, ga_df, mimic_df, xlim=None, ylim=None):
    fig, ax = plt.subplots()
    plt.grid()
    plt.title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness score")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # get fitness means and stdevs for each algorithm
    rhc_means, rhc_stdevs = means_stdevs(rhc_df)
    sa_means, sa_stdevs = means_stdevs(sa_df)
    ga_means, ga_stdevs = means_stdevs(ga_df)
    mimic_means, mimic_stdevs = means_stdevs(mimic_df)

    # shade in 1 stdev bounds for each algorithm
    ax.fill_between(rhc_df.index, rhc_means - rhc_stdevs, rhc_means + rhc_stdevs, color='b', alpha=0.1)
    ax.fill_between(sa_df.index, sa_means - sa_stdevs, sa_means + sa_stdevs, color='r', alpha=0.1)
    ax.fill_between(ga_df.index, ga_means - ga_stdevs, ga_means + ga_stdevs, color='g', alpha=0.1)
    ax.fill_between(mimic_df.index, mimic_means - mimic_stdevs, mimic_means + mimic_stdevs, color='y', alpha=0.1)

    # plot performance for each algorithm
    line_rhc = ax.plot(rhc_means, color='b', label="RHC")
    line_sa = ax.plot(sa_means, color='r', label="SA")
    line_ga = ax.plot(ga_means, color='g', label="GA")
    line_mimic = ax.plot(ga_means, color='y', label="MIMIC")

    lines = line_rhc + line_sa + line_ga + line_mimic
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, title="Algorithm", loc='best')
    plt.tight_layout()

    return plt


def compare_algos(problem_name, rhc_dfs, sa_dfs, ga_dfs, mimic_dfs):
    # collapse dfs to a single column w/ means across all runs
    mean_rhc_dfs = []
    mean_sa_dfs = []
    mean_ga_dfs = []
    mean_mimic_dfs = []

    for df in rhc_dfs: mean_rhc_dfs.append(means_stdevs(df)[0])
    for df in sa_dfs: mean_sa_dfs.append(means_stdevs(df)[0])
    for df in ga_dfs: mean_ga_dfs.append(means_stdevs(df)[0])
    for df in mimic_dfs: mean_mimic_dfs.append(means_stdevs(df)[0])

    # find the dataframe that performs the best for each algorithm
    best_rhc_fitness_vals = np.array([mean_df.max() for mean_df in mean_rhc_dfs])
    best_rhc_df = rhc_dfs[np.argmax(best_rhc_fitness_vals)]

    best_sa_fitness_vals = np.array([mean_df.max() for mean_df in mean_sa_dfs])
    best_sa_df = sa_dfs[np.argmax(best_sa_fitness_vals)]

    best_ga_fitness_vals = np.array([mean_df.max() for mean_df in mean_ga_dfs])
    best_ga_df = ga_dfs[np.argmax(best_ga_fitness_vals)]

    best_mimic_fitness_vals = np.array([mean_df.max() for mean_df in mean_mimic_dfs])
    best_mimic_df = mimic_dfs[np.argmax(best_mimic_fitness_vals)]

    plot_algo_comparisons(
        title="Algorithm comparison: fitness vs iterations",
        rhc_df=best_rhc_df,
        sa_df=best_sa_df,
        ga_df=best_ga_df,
        mimic_df=best_mimic_df,
    )

    plot_filename = 'graphs/' + problem_name + '_algo_comparison.png'
    plt.savefig(plot_filename)
    plt.clf()


if __name__ == "__main__":
    pass
