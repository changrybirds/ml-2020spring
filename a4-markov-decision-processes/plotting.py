import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_error_over_iters(error_df, title, file_path, xlim=None):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)

    if xlim is not None:
        ax1.set_xlim(0, xlim)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Error")
    line1 = ax1.plot(error_df, color='r', label="Error", marker='.')

    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(file_path)

    plt.clf()
    plt.close()
