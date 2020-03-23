import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_elbow_and_silhouette(sums_squared_distances, silhouette_scores, K_vals, title):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    ax1.set_xlabel("Num clusters")
    ax1.set_ylabel("Sum of squared distances")
    line1 = ax1.plot(K_vals, sums_squared_distances, color='r', label="Elbow method", marker='.')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette score")
    line2 = ax2.plot(K_vals, silhouette_scores, color='b', label="Silhouette score", marker='.')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.tight_layout()
    return plt


def plot_ll_and_bic(ll_scores, bic_scores, n_components_vals, title):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    ax1.set_xlabel("Num components")
    ax1.set_ylabel("Log likelihood")
    line1 = ax1.plot(n_components_vals, ll_scores, color='r', label="Log likelihood", marker='.')

    ax2 = ax1.twinx()
    ax2.set_ylabel("BIC score")
    line2 = ax2.plot(n_components_vals, bic_scores, color='b', label="BIC score", marker='.')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.tight_layout()
    return plt


def plot_cume_explained_variance(cume_explained_variances, n_components_vals, title):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    ax1.set_xlabel("Num components")
    ax1.set_ylabel("Cumulative explained variance")
    line1 = ax1.plot(
        n_components_vals, cume_explained_variances,
        color='r', label="Cumulative explained variance", marker='.')

    plt.tight_layout()
    return plt


def plot_kurtosis(kurtosis_vals, n_components_vals, title):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    ax1.set_xlabel("Num components")
    ax1.set_ylabel("Kurtosis")
    line1 = ax1.plot(
        n_components_vals, kurtosis_vals,
        color='r', label="Kurtosis", marker='.')

    plt.tight_layout()
    return plt


def plot_recon_loss(recon_errors, n_components_vals, title):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    ax1.set_xlabel("Num components")
    ax1.set_ylabel("Reconstruction error")
    line1 = ax1.plot(
        n_components_vals, recon_errors,
        color='r', label="Reconstruction error", marker='.')

    plt.tight_layout()
    return plt


def plot_model_complexity_charts(train_scores, test_scores, title, hp_name, ylim=None, xscale_type=None, basex=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(hp_name)
    plt.ylabel("Score")

    if ylim is not None:
        plt.ylim(*ylim)
    if xscale_type is not None:
        plt.xscale(xscale_type, basex=basex)
    plt.grid()

    plt.plot(train_scores, 'o-', color="r", label="Training score")
    plt.plot(test_scores, 'o-', color="g", label="CV score")
    plt.legend(loc='best')

    return plt


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
