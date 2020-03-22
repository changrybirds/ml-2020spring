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


def plot_recon_loss(recon_errors, iteration_vals, title):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reconstruction error")
    line1 = ax1.plot(
        iteration_vals, recon_errors,
        color='r', label="Reconstruction error", marker='.')

    plt.tight_layout()
    return plt
