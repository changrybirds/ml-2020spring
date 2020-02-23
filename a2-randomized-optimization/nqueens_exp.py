import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

import mlrose_hiive as mlrose

import plotting


# redefine n-queens as a maximization problem
# from tutorial docs: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
def queens_max(state):
    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

        # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt


def n_queens_rhc(nq_problem, initial_state, num_runs=20, verbose=False):
    runs = np.arange(num_runs)
    max_iters = 1000
    restarts = 10

    run_times = np.zeros(num_runs)
    fitness_data = pd.DataFrame()

    for run in runs:
        run_t0 = time()
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(
            problem=nq_problem,
            max_attempts=10,
            max_iters=max_iters,
            restarts=restarts,
            init_state=initial_state,
            curve=True,
        )
        run_time = time() - run_t0
        run_times[run] = run_time

        fitness_data = pd.concat([fitness_data, pd.DataFrame(fitness_curve)], axis=1, sort=False)
        # if verbose: print(fitness_data.tail())

    fitness_data.columns = runs
    fitness_data = fitness_data.fillna(method='ffill')

    avg_run_time = np.average(run_times)
    print("N-Queens - RHC avg run time:", avg_run_time)

    # generate plots
    plotting.plot_fitness_curves(
        fitness_data,
        title="RHC for N-queens: fitness vs. iterations",
    )

    plt.show()
    plt.savefig('graphs/nqueens_rhc_fitness.png')
    plt.clf()


def n_queens_sa(nq_problem, initial_state, num_runs=20, verbose=False):
    runs = np.arange(num_runs)
    max_iters = 1000
    decay_schedule = mlrose.ExpDecay()

    # if verbose: print(fitness_data.head())
    run_times = np.zeros(num_runs)
    fitness_data = pd.DataFrame()

    for run in runs:
        run_t0 = time()
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
            problem=nq_problem,
            schedule=decay_schedule,
            max_attempts=10,
            max_iters=max_iters,
            init_state=initial_state,
            curve=True,
            # random_state=313,
        )
        run_time = time() - run_t0
        run_times[run] = run_time

        # if verbose:
        #     print("Run", run, ':', fitness_curve[0:5])
        #     print("Fitness curve shape:", fitness_curve.shape)
        #     print(fitness_data.shape)

        fitness_data = pd.concat([fitness_data, pd.DataFrame(fitness_curve)], axis=1, sort=False)
        # if verbose: print(fitness_data.tail())

    fitness_data.columns = runs
    fitness_data = fitness_data.fillna(method='ffill')

    # print avg run time
    avg_run_time = np.average(run_times)
    print("N-Queens - SA avg run time:", avg_run_time)

    # generate plots
    plotting.plot_fitness_curves(
        fitness_data,
        title="SA for N-Queens: fitness vs. iterations",
    )

    plt.show()
    plt.savefig('graphs/nqueens_sa_fitness.png')
    plt.clf()


def main():
    verbose = True
    num_runs = 20

    # define custom fitness function to maximize instead of minimize
    fitness_fn = mlrose.CustomFitness(queens_max)

    # define optimization problem
    length = 16
    nq_problem = mlrose.DiscreteOpt(
        length=length,
        fitness_fn=fitness_fn,
        maximize=True,
        max_val=length,
    )

    # set initial state
    initial_state = np.random.randint(0, length, size=length)

    # randomized hill climbing
    n_queens_rhc(nq_problem, initial_state, num_runs, verbose=verbose)

    # simulated annealing
    n_queens_sa(nq_problem, initial_state, num_runs, verbose=verbose)


if __name__ == "__main__":
    main()
