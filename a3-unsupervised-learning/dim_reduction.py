import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kurtosis

from math import sqrt

import dataset_processing as data_proc
import plotting
import clustering

RANDOM_SEED = 313


def run_pca(dataset_name, X, y, verbose=False):
    # attempt PCA for various dimensionality levels
    n_components_vals = np.arange(1, len(X.columns))
    cume_explained_variances = []

    for n_components in n_components_vals:
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X)

        # calculate cumulative explained variance
        cume_explained_variance = np.sum(pca.explained_variance_)
        # if verbose: print(cume_explained_variance)
        cume_explained_variances.append(cume_explained_variance)

    # plot cumulative explained variance
    cume_explained_variances = np.array(cume_explained_variances)
    plot_title = "PCA for " + dataset_name + ": Cume. explained variance\n"
    plotting.plot_cume_explained_variance(
        cume_explained_variances, n_components_vals, title=plot_title)
    plt.savefig('graphs/pca_' + dataset_name + '_cumevar.png')
    plt.clf()

    # choose optimal number of components (clusters) based on max cumulative explained variance
    if dataset_name == 'abalone':
        optimal_comp = 3
    else:
        optimal_comp = 25
    opt_pca = PCA(n_components=optimal_comp, random_state=RANDOM_SEED)
    opt_X_pca = opt_pca.fit_transform(X)

    # calculate reconstruction loss
    X_projected = opt_pca.inverse_transform(opt_X_pca)
    recon_loss = ((X - X_projected) ** 2).mean()
    print(dataset_name, ": PCA reconstruction loss for k =", optimal_comp, ":", np.sum(recon_loss), '\n')
    opt_X_pca = pd.DataFrame(opt_X_pca)

    # run K-means
    clustering.run_k_means(dataset_name, opt_X_pca, y, dim_reduction='pca', verbose=verbose)

    # run EM
    clustering.run_expect_max(dataset_name, opt_X_pca, y, dim_reduction='pca', verbose=verbose)

    return opt_X_pca


def run_ica(dataset_name, X, y, verbose=False):
    # attempt ICA for various dimensionality levels
    n_components_vals = np.arange(1, len(X.columns))
    kurtosis_vals = []

    for n_components in n_components_vals:
        ica = FastICA(n_components=n_components, random_state=RANDOM_SEED)
        X_ica = ica.fit_transform(X)

        # calculate cumulative explained variance
        kurtosis_val = np.average(kurtosis(ica.components_, fisher=False))
        # if verbose: print(kurtosis_val)
        kurtosis_vals.append(kurtosis_val)

    # plot cumulative explained variance
    kurtosis_vals = np.array(kurtosis_vals)
    plot_title = "ICA for " + dataset_name + ": Kurtosis\n"
    plotting.plot_kurtosis(
        kurtosis_vals, n_components_vals, title=plot_title)
    plt.savefig('graphs/ica_' + dataset_name + '_kurtosis.png')
    plt.clf()

    # choose optimal number of components (clusters) based on max cumulative explained variance
    if dataset_name == 'abalone':
        optimal_comp = 9
    else:
        optimal_comp = 56
    opt_ica = FastICA(n_components=optimal_comp, random_state=RANDOM_SEED)
    opt_X_ica = opt_ica.fit_transform(X)

    # calculate reconstruction loss
    X_projected = opt_ica.inverse_transform(opt_X_ica)
    recon_loss = ((X - X_projected) ** 2).mean()
    print(dataset_name, ": ICA reconstruction loss for k =", optimal_comp, ":", np.sum(recon_loss), '\n')
    opt_X_ica = pd.DataFrame(opt_X_ica)

    # run K-means
    clustering.run_k_means(dataset_name, opt_X_ica, y, dim_reduction='ica', verbose=verbose)

    # run EM
    clustering.run_expect_max(dataset_name, opt_X_ica, y, dim_reduction='ica', verbose=verbose)

    return opt_X_ica


def run_rp(dataset_name, X, y, verbose=False):
    # attempt RP for various dimensionality levels
    n_components_vals = np.arange(1, len(X.columns))
    iterations = np.arange(1, 15)
    recon_losses = []

    for n_components in n_components_vals:
        # see how reconstruction loss changes across iterations
        tmp_recon_losses = []
        for i in iterations:
            rp = GaussianRandomProjection(n_components=n_components, random_state=i)
            X_rp = rp.fit_transform(X)

            # calculate reconstruction error
            X_comp_pinv = np.linalg.pinv(rp.components_.T)
            X_projection = np.dot(X_rp, X_comp_pinv)
            recon_loss = ((X - X_projection) ** 2).mean()
            # if verbose: print(recon_loss.shape)
            tmp_recon_losses.append(np.sum(recon_loss))

        tmp_avg_recon_loss = np.mean(np.array(tmp_recon_losses))
        recon_losses.append(tmp_avg_recon_loss)

    if dataset_name == 'abalone':
        n_components = 3
    else:
        n_components = 25

    # plot reconstruction losses
    # if verbose: print(recon_losses[0])
    recon_losses = np.array(recon_losses)
    plot_title = "RP for " + dataset_name + ": Reconstruction loss\n"
    plotting.plot_recon_loss(
        recon_losses, n_components_vals, title=plot_title)
    plt.savefig('graphs/rp_' + dataset_name + '_recon_loss.png')
    plt.clf()

    # calculate reconstruction error
    grp = GaussianRandomProjection(n_components=n_components, random_state=RANDOM_SEED)
    X_rp = grp.fit_transform(X)

    X_comp_pinv = np.linalg.pinv(grp.components_.T)
    X_projection = np.dot(X_rp, X_comp_pinv)
    recon_loss = ((X - X_projection) ** 2).mean()

    print(dataset_name, ": RP reconstruction loss for k =", n_components, ":", np.sum(recon_loss), '\n')
    X_rp = pd.DataFrame(X_rp)

    # run K-means
    clustering.run_k_means(dataset_name, X_rp, y, dim_reduction='rp', verbose=verbose)

    # run EM
    clustering.run_expect_max(dataset_name, X_rp, y, dim_reduction='rp', verbose=verbose)

    return X_rp


def run_dt_fi(dataset_name, X, y, verbose=False):
    dtclf = DecisionTreeClassifier(random_state=RANDOM_SEED)
    dtclf.fit(X, y)
    fi = dtclf.feature_importances_

    fi_df = pd.DataFrame(fi, index=X.columns, columns=['feature_importance'])
    fi_df = fi_df.sort_values('feature_importance', ascending=False)
    # if verbose: print(fi_df)

    csv_path = 'tmp/dt_fi_' + dataset_name + '.csv'
    fi_df.to_csv(csv_path, header=True)

    # slice top n features
    if dataset_name == 'abalone':
        num_features = 3
    else:
        num_features = 25

    selected_features = fi_df.index[0:num_features].tolist()
    X_selected = X[selected_features]

    print("-------- DT_FI complete! --------\n")

    # run K-means
    clustering.run_k_means(dataset_name, X_selected, y, dim_reduction='dt_fi', verbose=verbose)

    # run EM
    clustering.run_expect_max(dataset_name, X_selected, y, dim_reduction='dt_fi', verbose=verbose)

    return X_selected


def abalone(verbose=False):
    X, y = data_proc.process_abalone(scaler='minmax', tt_split=False)
    run_pca('abalone', X, y, verbose=verbose)
    run_ica('abalone', X, y, verbose=verbose)
    run_rp('abalone', X, y, verbose=verbose)
    run_dt_fi('abalone', X, y, verbose=verbose)


def online_shopping(verbose=False):
    X, y = data_proc.process_online_shopping(scaler='minmax', tt_split=False)
    run_pca('shopping', X, y, verbose=verbose)
    run_ica('shopping', X, y, verbose=verbose)
    run_rp('shopping', X, y, verbose=verbose)
    run_dt_fi('shopping', X, y, verbose=verbose)


def main():
    abalone(verbose=True)
    online_shopping(verbose=True)


if __name__ == "__main__":
    main()
