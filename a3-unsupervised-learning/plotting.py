import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_elbow_and_silhouette(sums_squared_distances, silhouette_scores, K_vals, title):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    ax1.set_xlabel("Num clusters")
    ax1.set_ylabel("Sum of squared distances")
    line1 = ax1.plot(K_vals, sums_squared_distances, color='r', label="Elbow method")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette score")
    line2 = ax2.plot(K_vals, silhouette_scores, color='b', label="Silhouette score")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.tight_layout()
    return plt