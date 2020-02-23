import numpy as np
import pandas as pd

from time import time

import mlrose_hiive as mlrose


def simulated_annealing(problem, init_state, schedule, max_attempts=10, max_iters=1000, curve=True, num_runs=10):
    # run SA
    runs = np.arange(num_runs)
    max_iters = max_iters
    fitness_data = pd.DataFrame()
    # if verbose: print(fitness_data.head())
    run_times = np.zeros(num_runs)

    for run in runs:
        run_t0 = time()
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
            problem,
            schedule=schedule,
            max_attempts=max_attempts,
            max_iters=max_iters,
            init_state=init_state,
            curve=curve,
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

    return fitness_data, run_times
