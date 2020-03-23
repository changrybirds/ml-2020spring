import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt

from time import time

import dataset_processing as data_proc
import clustering
import dim_reduction
import plotting

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def model_complexity_curve(X_train, y_train, max_iter, hp, hp_vals, cv=None):
    # note: this assumes we use 1 hidden layer
    hidden_units = []
    for val in hp_vals:
        if type(val) is tuple:
            hidden_units.append(val[0])
        else:
            hidden_units.append(val)

    df = pd.DataFrame(index=hidden_units, columns=['train', 'cv'])

    for hp_val, hu_num in zip(hp_vals, hidden_units):
        kwargs = {
            hp: hp_val,
            'max_iter': max_iter,
            'random_state': data_proc.SEED_VAL}

        mlpclf = MLPClassifier(**kwargs)

        # train data
        mlpclf.fit(X_train, y_train)
        train_score = mlpclf.score(X_train, y_train)

        # get cv scores
        cross_vals = cross_val_score(mlpclf, X_train, y_train, cv=cv)
        cv_mean = np.mean(cross_vals)

        df.loc[hu_num, 'train'] = train_score
        df.loc[hu_num, 'cv'] = cv_mean

    return pd.DataFrame(df, dtype='float')


@ignore_warnings(category=ConvergenceWarning)
def nn_iterative_lc(X, y, max_iter_range, cv=None):
    df = pd.DataFrame(index=max_iter_range, columns=['train', 'cv', 'train_time', 'cv_time'])
    for i in max_iter_range:
        kwargs = {
            'max_iter': i,
            'random_state': data_proc.SEED_VAL}

        mlpclf = MLPClassifier(**kwargs)

        # train data
        train_t0 = time()
        mlpclf.fit(X, y)
        train_time = time() - train_t0
        train_score = mlpclf.score(X, y)

        # get cv scores
        cv_t0 = time()
        cross_vals = cross_val_score(mlpclf, X, y, cv=cv)
        cv_time = time() - cv_t0
        cv_mean = np.mean(cross_vals)

        df.loc[i, 'train'] = train_score
        df.loc[i, 'cv'] = cv_mean
        df.loc[i, 'train_time'] = train_time
        df.loc[i, 'cv_time'] = cv_time

    return df.astype('float64')


@ignore_warnings(category=ConvergenceWarning)
def run_experiment(dataset_name, exp_name, X_train, X_test, y_train, y_test, verbose=False, show_plots=False):
    # calculate and print learning curves, use max_iter as x-axis
    max_iter_range = np.arange(100, 500, 50)

    lc_df = nn_iterative_lc(X_train, y_train, max_iter_range, cv=data_proc.CV_VAL)
    if verbose: print(lc_df.head(10))

    max_iter_hp = lc_df['cv'].idxmax()
    if verbose: print(lc_df.idxmax())

    plotting.plot_iterative_lc(
        lc_df, dataset_name + ' (' + exp_name + '): learning curves (iterations)', max_iter_range=max_iter_range)

    if show_plots:
        plt.show()

    plt.savefig('graphs/nn_lc_' + exp_name + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # calculate model complexity scores for hidden_layer_sizes
    num_features = X_train.shape[1]
    hp = 'hidden_layer_sizes'

    # using 1 hidden layer - problems shouldn't be so complex as to require more
    features = range(1, num_features)
    hp_vals = []
    for feature in features:
        hp_vals.append((feature,))

    if verbose: print(hp_vals)
    hidden_layer_sizes_mc = model_complexity_curve(
        X_train, y_train, max_iter_hp, hp, hp_vals, cv=data_proc.CV_VAL)
    hidden_layer_sizes_hp = hidden_layer_sizes_mc['cv'].idxmax()

    if verbose: print(hidden_layer_sizes_mc.head(10))
    if verbose: print(hidden_layer_sizes_mc.idxmax())

    plotting.plot_model_complexity_charts(
        hidden_layer_sizes_mc['train'], hidden_layer_sizes_mc['cv'],
        dataset_name + ' (' + exp_name + '): MCC for ' + hp, hp)

    if show_plots:
        plt.show()

    plt.savefig('graphs/nn_mcc_' + exp_name + '_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # calculate model complexity scores for learning_rate_init
    hp = 'learning_rate_init'
    hp_vals = np.logspace(-5, 0, base=10.0, num=6)  # this should vary for each hyperparameter
    learning_rate_init_mc = model_complexity_curve(
        X_train, y_train, max_iter_hp, hp, hp_vals, cv=data_proc.CV_VAL)
    learning_rate_init_hp = learning_rate_init_mc['cv'].idxmax()

    if verbose: print(learning_rate_init_mc.head(10))
    if verbose: print(learning_rate_init_mc.idxmax())

    plotting.plot_model_complexity_charts(
        learning_rate_init_mc['train'], learning_rate_init_mc['cv'],
        dataset_name + ' (' + exp_name + '): MCC for ' + hp, hp, xscale_type='log', basex=10)

    if show_plots:
        plt.show()

    plt.savefig('graphs/nn_mcc_' + exp_name + '_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # instantiate neural network
    mlpclf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes_hp, learning_rate_init=learning_rate_init_hp,
        max_iter=max_iter_hp, random_state=data_proc.SEED_VAL)

    mlpclf.fit(X_train, y_train)

    train_score = data_proc.model_train_score(mlpclf, X_train, y_train)
    test_score = data_proc.model_test_score(mlpclf, X_test, y_test)
    print("MLPClassifier training set score for", exp_name, dataset_name + ": ", train_score)
    print("MLPClassifier holdout set score for", exp_name, dataset_name + ": ", test_score)


def abalone_dr(verbose=False, show_plots=False):
    X, y = data_proc.process_abalone(scaler='minmax', tt_split=False)

    # calculate baseline performance
    base_X_train, base_X_test, base_y_train, base_y_test = data_proc.process_abalone(tt_split=True)
    run_experiment(
        'abalone', 'baseline',
        base_X_train, base_X_test, base_y_train, base_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nBaseline complete!\n", "------------------------------\n")

    # calculate PCA performance
    pca_X = dim_reduction.run_pca('abalone', X, y, verbose=verbose)
    pca_X_train, pca_X_test, pca_y_train, pca_y_test = train_test_split(pca_X, y)
    run_experiment(
        'abalone', 'pca',
        pca_X_train, pca_X_test, pca_y_train, pca_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nPCA complete!\n", "------------------------------\n")

    # calculate ICA performance
    ica_X = dim_reduction.run_ica('abalone', X, y, verbose=verbose)
    ica_X_train, ica_X_test, ica_y_train, ica_y_test = train_test_split(ica_X, y)
    run_experiment(
        'abalone', 'ica',
        ica_X_train, ica_X_test, ica_y_train, ica_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nICA complete!\n", "------------------------------\n")

    # calculate RP performance
    rp_X = dim_reduction.run_rp('abalone', X, y, verbose=verbose)
    rp_X_train, rp_X_test, rp_y_train, rp_y_test = train_test_split(rp_X, y)
    run_experiment(
        'abalone', 'rp',
        rp_X_train, rp_X_test, rp_y_train, rp_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nRP complete!\n", "------------------------------\n")

    # calculate DT_FI performance
    dt_fi_X = dim_reduction.run_dt_fi('abalone', X, y, verbose=verbose)
    dt_fi_X_train, dt_fi_X_test, dt_fi_y_train, dt_fi_y_test = train_test_split(dt_fi_X, y)
    run_experiment(
        'abalone', 'dt_fi',
        dt_fi_X_train, dt_fi_X_test, dt_fi_y_train, dt_fi_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nDT_FI complete!\n", "------------------------------\n")


def abalone_cluster(verbose=False, show_plots=False):
    X, y = data_proc.process_abalone(scaler='minmax', tt_split=False)

    # calculate baseline performance
    base_X_train, base_X_test, base_y_train, base_y_test = data_proc.process_abalone(tt_split=True)
    run_experiment(
        'abalone', 'baseline',
        base_X_train, base_X_test, base_y_train, base_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nBaseline complete!\n", "------------------------------\n")

    # calculate K-means performance
    k_means_X_clusters = clustering.run_k_means('abalone', X, y, dim_reduction=None, verbose=verbose)
    k_means_X_train, k_means_X_test, k_means_y_train, k_means_y_test = data_proc.process_abalone_w_clusters(
        k_means_X_clusters, scaler='minmax')

    run_experiment(
        'abalone', 'kmeans',
        k_means_X_train, k_means_X_test, k_means_y_train, k_means_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nK-means complete!\n", "------------------------------\n")

    # calculate Expectation Maximization performance
    em_X_clusters = clustering.run_expect_max('abalone', X, y, dim_reduction=None, verbose=verbose)
    em_X_train, em_X_test, em_y_train, em_y_test = data_proc.process_abalone_w_clusters(
        em_X_clusters, scaler='minmax')

    run_experiment(
        'abalone', 'em',
        em_X_train, em_X_test, em_y_train, em_y_test,
        verbose=verbose, show_plots=show_plots,
    )
    if verbose: print("\nExpectation Maximization complete!\n", "------------------------------\n")


def main():
    abalone_dr(verbose=True, show_plots=False)
    abalone_cluster(verbose=True, show_plots=False)


if __name__ == "__main__":
    main()
