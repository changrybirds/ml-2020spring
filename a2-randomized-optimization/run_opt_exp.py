import numpy as np
import pandas as pd

import mlrose_hiive as mlrose


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


def n_queens_exp(verbose=False):
    fitness_fn = mlrose.CustomFitness(queens_max)  # define fitness function

    # define optimization problem
    nq_problem = mlrose.DiscreteOpt(
        length=8,
        fitness_fn=fitness_fn,
        maximize=True,
        max_val=8,
    )

    # set initial state
    initial_state = np.random.randint(0, 8, size=8)
    if verbose: print(initial_state)

    # set decay schedule for SA
    sa_decay_schedule = mlrose.ExpDecay()

    # run SA
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
        nq_problem,
        schedule=sa_decay_schedule,
        max_attempts=10,
        max_iters=1000,
        init_state=initial_state,
        curve=True,
        random_state=313,
    )

    if verbose:
        print(best_state)
        print(best_fitness)
        print(fitness_curve)


def flipflop_exp():
    pass


def tsm_experiment():
    pass


def main():
    n_queens_exp(verbose=True)


if __name__ == "__main__":
    main()
