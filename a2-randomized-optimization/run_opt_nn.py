import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from time import time

import mlrose_hiive as mlrose

import dataset_proc as data_proc
import plotting_nn

HIDDEN_NODES = (9,)
LEARNING_RATE = .001
MAX_ITERS = 100
RANDOM_SEED_VAL = 313
NUM_RUNS = 10
CV_VAL = 5


def nn_iterative_lc(X, y, max_iter_range, kwargs, cv=None):
    df = pd.DataFrame(index=max_iter_range, columns=['train', 'cv', 'train_time', 'cv_time'])
    for i in max_iter_range:
        kwargs['max_iters'] = i.item()
        mlr_nn = mlrose.NeuralNetwork(**kwargs)

        # train data
        train_t0 = time()
        mlr_nn.fit(X, y)
        train_time = time() - train_t0
        train_score = mlr_nn.score(X, y)

        # get cv scores
        cv_t0 = time()
        cross_vals = cross_val_score(mlr_nn, X, y, cv=CV_VAL)
        cv_time = time() - cv_t0
        cv_mean = np.mean(cross_vals)

        df.loc[i, 'train'] = train_score
        df.loc[i, 'cv'] = cv_mean
        df.loc[i, 'train_time'] = train_time
        df.loc[i, 'cv_time'] = cv_time

    return df.astype('float64')


def gradient_descent(X_train, X_test, y_train, y_test, verbose=False):
    max_iter_range = np.arange(50, 500, 50)
    kwargs = {
        'hidden_nodes': HIDDEN_NODES,
        'activation': 'relu',
        'learning_rate': LEARNING_RATE,
        'random_state': RANDOM_SEED_VAL,
        'curve': True,

        # algorithm-specific
        'algorithm': 'gradient_descent',
    }

    nn_gd = mlrose.NeuralNetwork(**kwargs)

    nn_gd.fit(X_train, y_train)

    # plot fitness curve
    plot_title = "NN weight opt - GD: Ffitness vs. iterations"
    plotting_nn.plot_fitness_curves(
        fitness_data=pd.DataFrame(nn_gd.fitness_curve),
        title=plot_title,
    )
    plt.savefig('graphs/nn_gd_fitness_curve.png')
    plt.clf()

    # plot iterative learning curve
    lc_df = nn_iterative_lc(X_train, y_train, max_iter_range, kwargs)
    plotting_nn.plot_iterative_lc(
        lc_df,
        title="Learning Curve (max_iters) for GD",
        max_iter_range=max_iter_range,
    )
    plt.savefig('graphs/nn_gd_lc_iterations.png')
    plt.clf()

    # plot learning curve
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(
        nn_gd,
        title="Learning curve for GD",
        X=X_train,
        y=y_train,
        cv=CV_VAL,
        train_sizes=train_sizes,
    )
    plt.savefig('graphs/nn_gd_lc.png')
    plt.clf()


def random_hill_climbing(X_train, X_test, y_train, y_test, verbose=False):
    max_iter_range = np.arange(50, 500, 50)
    kwargs = {
        'hidden_nodes': HIDDEN_NODES,
        'activation': 'relu',
        'learning_rate': LEARNING_RATE,
        'random_state': RANDOM_SEED_VAL,
        'curve': True,

        # algorithm-specific
        'algorithm': 'random_hill_climb',
    }

    nn_rhc = mlrose.NeuralNetwork(**kwargs)

    nn_rhc.fit(X_train, y_train)

    # plot fitness curve
    plot_title = "NN weight opt - RHC: Ffitness vs. iterations"
    plotting_nn.plot_fitness_curves(
        fitness_data=pd.DataFrame(nn_rhc.fitness_curve),
        title=plot_title,
    )
    plt.savefig('graphs/nn_rhc_fitness_curve.png')
    plt.clf()

    # plot iterative learning curve
    lc_df = nn_iterative_lc(X_train, y_train, max_iter_range, kwargs)
    plotting_nn.plot_iterative_lc(
        lc_df,
        title="Learning Curve (max_iters) for RHC",
        max_iter_range=max_iter_range,
    )
    plt.savefig('graphs/nn_rhc_lc_iterations.png')
    plt.clf()

    # plot learning curve
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(
        nn_rhc,
        title="Learning curve for RHC",
        X=X_train,
        y=y_train,
        cv=CV_VAL,
        train_sizes=train_sizes,
    )
    plt.savefig('graphs/nn_rhc_lc.png')
    plt.clf()


def simulated_annealing(X_train, X_test, y_train, y_test, verbose=False):
    max_iter_range = np.arange(50, 500, 50)
    kwargs = {
        'hidden_nodes': HIDDEN_NODES,
        'activation': 'relu',
        'learning_rate': LEARNING_RATE,
        'random_state': RANDOM_SEED_VAL,
        'curve': True,

        # algorithm-specific
        'algorithm': 'simulated_annealing',
    }

    nn_sa = mlrose.NeuralNetwork(**kwargs)

    nn_sa.fit(X_train, y_train)

    # plot fitness curve
    plot_title = "NN weight opt - SA: Ffitness vs. iterations"
    plotting_nn.plot_fitness_curves(
        fitness_data=pd.DataFrame(nn_sa.fitness_curve),
        title=plot_title,
    )
    plt.savefig('graphs/nn_sa_fitness_curve.png')
    plt.clf()

    # plot iterative learning curve
    lc_df = nn_iterative_lc(X_train, y_train, max_iter_range, kwargs)
    plotting_nn.plot_iterative_lc(
        lc_df,
        title="Learning Curve (max_iters) for SA",
        max_iter_range=max_iter_range,
    )
    plt.savefig('graphs/nn_sa_lc_iterations.png')
    plt.clf()

    # plot learning curve
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(
        nn_sa,
        title="Learning curve for SA",
        X=X_train,
        y=y_train,
        cv=CV_VAL,
        train_sizes=train_sizes,
    )
    plt.savefig('graphs/nn_sa_lc.png')
    plt.clf()


def genetic_algorithm(X_train, X_test, y_train, y_test, verbose=False):
    max_iter_range = np.arange(50, 500, 50)
    kwargs = {
        'hidden_nodes': HIDDEN_NODES,
        'activation': 'relu',
        'learning_rate': LEARNING_RATE,
        'random_state': RANDOM_SEED_VAL,
        'curve': True,

        # algorithm-specific
        'algorithm': 'genetic_alg',

    }

    nn_ga = mlrose.NeuralNetwork(**kwargs)

    nn_ga.fit(X_train, y_train)

    # plot fitness curve
    plot_title = "NN weight opt - GA: Ffitness vs. iterations"
    plotting_nn.plot_fitness_curves(
        fitness_data=pd.DataFrame(nn_ga.fitness_curve),
        title=plot_title,
    )
    plt.savefig('graphs/nn_ga_fitness_curve.png')
    plt.clf()

    # plot iterative learning curve
    lc_df = nn_iterative_lc(X_train, y_train, max_iter_range, kwargs)
    plotting_nn.plot_iterative_lc(
        lc_df,
        title="Learning Curve (max_iters) for GA",
        max_iter_range=max_iter_range,
    )
    plt.savefig('graphs/nn_ga_lc_iterations.png')
    plt.clf()

    # plot learning curve
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(
        nn_ga,
        title="Learning curve for GA",
        X=X_train,
        y=y_train,
        cv=CV_VAL,
        train_sizes=train_sizes,
    )
    plt.savefig('graphs/nn_ga_lc.png')
    plt.clf()


def main():
    X_train, X_test, y_train, y_test = data_proc.process_abalone()

    # gradient_descent(X_train, X_test, y_train, y_test, verbose=True)

    # random_hill_climbing(X_train, X_test, y_train, y_test, verbose=True)

    # simulated_annealing(X_train, X_test, y_train, y_test, verbose=True)

    genetic_algorithm(X_train, X_test, y_train, y_test, verbose=True)


if __name__ == "__main__":
    main()
