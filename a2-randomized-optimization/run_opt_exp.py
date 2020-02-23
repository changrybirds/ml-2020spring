import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

import mlrose_hiive as mlrose

import plotting
import opt_algos


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


def n_queens_exp(verbose=False, num_runs=20):
    fitness_fn = mlrose.CustomFitness(queens_max)  # define fitness function

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
    # if verbose: print(initial_state)

    # set decay schedule for SA and begin runs
    decay_schedule = mlrose.ExpDecay()
    fitness_data, run_times = opt_algos.simulated_annealing(
        problem=nq_problem,
        init_state=initial_state,
        schedule=decay_schedule,
        max_attempts=10,
        max_iters=1000,
        curve=True,
        num_runs=num_runs,
    )

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

    # if verbose: print(fitness_data.tail())


def flipflop_exp():
    pass


def tsm_experiment():
    pass


def main():
    n_queens_exp(verbose=True, num_runs=20)


if __name__ == "__main__":
    main()
