import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import dataset_processing as data_proc
import plotting

RANDOM_SEED = 313


def run_k_means(dataset_name, X, y, dim_reduction=None, verbose=False):
    # find optimal number of clusters
    K_vals = np.arange(2, len(X.columns))
    sums_squared_distances = []
    silhouette_scores = []

    for k in K_vals:
        k_means = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        k_means.fit(X)
        y_hat = k_means.predict(X)
        labels = k_means.labels_

        # calculate sum of squared distances and silhouette score
        sums_squared_distances.append(k_means.inertia_)
        s_score = silhouette_score(X, labels, metric='euclidean')
        silhouette_scores.append(s_score)

    # plot elbow and silhouette scores
    sums_squared_distances = np.array(sums_squared_distances)
    silhouette_scores = np.array(silhouette_scores)

    if dim_reduction is None: dim_reduction = 'no_dr'
    plot_title = "K-means w/ " + dim_reduction + " for " + dataset_name + ": Elbow plot & silhouette scores\n"
    plotting.plot_elbow_and_silhouette(
        sums_squared_distances, silhouette_scores, K_vals, title=plot_title)
    plt.savefig('graphs/kmeans_' + dim_reduction + '_' + dataset_name + '_elbow_silhouette.png')
    plt.clf()

    # choose optimal number of clusters
    optimal_k = K_vals[np.argmax(silhouette_scores)]
    if verbose: print(optimal_k)

    opt_k_means = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED)
    opt_k_means.fit(X)
    opt_y_hat = opt_k_means.predict(X)
    opt_labels = opt_k_means.labels_

    # using optimal no. clusters, examine label distribution - no dimensionality reduction
    df = X.copy(deep=True)
    df['y'] = y
    df['cluster'] = opt_labels
    grouped = df.groupby(['y', 'cluster']).count().iloc[:, 0]

    csv_path = 'tmp/kmeans_' + dim_reduction + '_' + dataset_name + '.csv'
    grouped.to_csv(csv_path, header=True)
    if verbose: print(dataset_name, '\n', grouped, '\n')


def run_expect_max(dataset_name, X, y, dim_reduction=None, verbose=False):
    # find optimal number of components
    n_components_vals = np.arange(2, len(X.columns))
    ll_scores = []
    bic_scores = []

    for n_components in n_components_vals:
        gm = GaussianMixture(n_components=n_components, random_state=RANDOM_SEED)
        gm.fit(X)
        y_hat = gm.predict(X)

        # calculate log likelihood scores and BIC scores
        ll_scores.append(gm.score(X))
        bic_scores.append(gm.bic(X))

    # plot BIC scores
    bic_scores = np.array(bic_scores)

    if dim_reduction is None: dim_reduction = 'no_dr'
    plot_title = "EM w/ " + dim_reduction + " for " + dataset_name + ": Log likelihood & BIC\n"
    plotting.plot_ll_and_bic(
        ll_scores, bic_scores, n_components_vals, title=plot_title)
    plt.savefig('graphs/em_' + dim_reduction + '_' + dataset_name + '_bic.png')
    plt.clf()

    # choose optimal number of components (clusters)
    optimal_comp = n_components_vals[np.argmin(bic_scores)]
    if verbose: print(optimal_comp)

    opt_em = GaussianMixture(n_components=optimal_comp, random_state=RANDOM_SEED)
    opt_em.fit(X)
    opt_y_hat = opt_em.predict(X)

    # using optimal no. clusters, examine label distribution - no dimensionality reduction
    df = X.copy(deep=True)
    df['y'] = y
    df['cluster'] = opt_y_hat
    grouped = df.groupby(['y', 'cluster']).count().iloc[:, 0]

    csv_path = 'tmp/em_' + dim_reduction + '_' + dataset_name + '.csv'
    grouped.to_csv(csv_path, header=True)
    if verbose: print(dataset_name, '\n', grouped, '\n')


def abalone(verbose=False):
    X, y = data_proc.process_abalone(scaler='minmax')
    run_k_means('abalone', X, y, verbose=verbose)
    run_expect_max('abalone', X, y, verbose=verbose)


def online_shopping(verbose=False):
    X, y = data_proc.process_online_shopping(scaler='minmax')
    run_k_means('shopping', X, y, verbose=verbose)
    run_expect_max('shopping', X, y, verbose=verbose)


def main():
    abalone(verbose=True)
    online_shopping(verbose=True)


if __name__ == "__main__":
    main()
