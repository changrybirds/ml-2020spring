import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

import mlrose_hiive as mlrose

import plotting


# redefine n-queens as a maximization problem
# adapted from tutorial docs: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
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


def n_queens_rhc(nq_problem, initial_state, max_iters=np.inf, num_runs=20, verbose=False):
    hp_name = 'restarts'
    hp_values = [10, 20, 30]

    # run for each hp value and append results to list

    fitness_dfs = []
    runs = np.arange(num_runs)

    for hp_value in hp_values:
        restarts = hp_value  # set varied HP at beginning of loop

        run_times = np.zeros(num_runs)
        fitness_data = pd.DataFrame()

        for run in runs:
            run_t0 = time()
            best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(
                problem=nq_problem,
                restarts=restarts,
                max_attempts=10,
                max_iters=max_iters,
                init_state=initial_state,
                curve=True,
            )
            run_time = time() - run_t0
            run_times[run] = run_time

            fitness_data = pd.concat([fitness_data, pd.DataFrame(fitness_curve)], axis=1, sort=False)

        fitness_data.columns = runs
        fitness_data = fitness_data.fillna(method='ffill')
        fitness_dfs.append(fitness_data)

        # calculate and print avg time per run
        avg_run_time = np.average(run_times)
        print("N-Queens - RHC avg run time,", hp_value, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "N-Queens RHC: fitness vs. iterations"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/n_queens_rhc_fitness.png')
    plt.clf()

    return fitness_dfs


def n_queens_sa(nq_problem, initial_state, max_iters=np.inf, num_runs=20, verbose=False):
    hp_name = 'schedule'
    hp_values = [mlrose.ArithDecay(), mlrose.GeomDecay(), mlrose.ExpDecay()]
    hp_values_strings = [val.get_info__()['schedule_type'] for val in hp_values]

    # run for each hp value and append results to list

    fitness_dfs = []
    runs = np.arange(num_runs)

    for hp_value, hp_value_string in zip(hp_values, hp_values_strings):
        schedule = hp_value  # set varied HP at beginning of loop

        run_times = np.zeros(num_runs)
        fitness_data = pd.DataFrame()

        for run in runs:
            run_t0 = time()
            best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
                problem=nq_problem,
                schedule=schedule,
                max_attempts=10,
                max_iters=max_iters,
                curve=True,
            )
            run_time = time() - run_t0
            run_times[run] = run_time

            fitness_data = pd.concat([fitness_data, pd.DataFrame(fitness_curve)], axis=1, sort=False)

        fitness_data.columns = runs
        fitness_data = fitness_data.fillna(method='ffill')
        fitness_dfs.append(fitness_data)

        # calculate and print avg time per run
        avg_run_time = np.average(run_times)
        print("N-Queens - SA avg run time,", hp_value_string, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "N-Queens SA: fitness vs. iterations"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values_strings,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/n_queens_sa_fitness.png')
    plt.clf()

    return fitness_dfs


def n_queens_ga(nq_problem, max_iters=np.inf, num_runs=20, verbose=False):
    # HP to vary
    hp_name = 'pop_mate_pct'
    hp_values = [0.25, 0.50, 0.75]

    # other hyperparameters for genetic algorithm
    population_size = 200
    elite_dreg_ratio = 0.95
    mutation_prob = 0.1

    # run for each hp value and append results to list

    fitness_dfs = []
    runs = np.arange(num_runs)

    for hp_value in hp_values:
        pop_mate_pct = hp_value  # set varied HP at beginning of loop

        run_times = np.zeros(num_runs)
        fitness_data = pd.DataFrame()

        for run in runs:
            run_t0 = time()
            best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
                problem=nq_problem,
                pop_size=population_size,
                pop_breed_percent=pop_mate_pct,
                elite_dreg_ratio=elite_dreg_ratio,
                mutation_prob=mutation_prob,
                max_attempts=10,
                max_iters=max_iters,
                curve=True,
            )
            run_time = time() - run_t0
            run_times[run] = run_time

            fitness_data = pd.concat([fitness_data, pd.DataFrame(fitness_curve)], axis=1, sort=False)

        fitness_data.columns = runs
        fitness_data = fitness_data.fillna(method='ffill')
        fitness_dfs.append(fitness_data)

        # calculate and print avg time per run
        avg_run_time = np.average(run_times)
        print("N-Queens - GA avg run time,", hp_value, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "N-Queens GA - " \
        + str(population_size) + " pop, " \
        + str(mutation_prob) + " mut prob, " \
        + ": fit vs iter"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/n_queens_ga_fitness.png')
    plt.clf()

    return fitness_dfs


def n_queens_mimic(nq_problem, max_iters=np.inf, num_runs=20, verbose=False):
    # HP to vary
    hp_name = 'keep_pct'
    hp_values = [0.2, 0.4, 0.6]

    # other hyperparameters for genetic algorithm
    population_size = 200

    # run for each hp value and append results to list

    fitness_dfs = []
    runs = np.arange(num_runs)

    for hp_value in hp_values:
        keep_pct = hp_value  # set varied HP at beginning of loop

        run_times = np.zeros(num_runs)
        fitness_data = pd.DataFrame()

        for run in runs:
            run_t0 = time()
            best_state, best_fitness, fitness_curve = mlrose.mimic(
                problem=nq_problem,
                pop_size=population_size,
                keep_pct=keep_pct,
                max_attempts=10,
                max_iters=max_iters,
                curve=True,
            )
            run_time = time() - run_t0
            run_times[run] = run_time

            fitness_data = pd.concat([fitness_data, pd.DataFrame(fitness_curve)], axis=1, sort=False)

        fitness_data.columns = runs
        fitness_data = fitness_data.fillna(method='ffill')
        fitness_dfs.append(fitness_data)

        # calculate and print avg time per run
        avg_run_time = np.average(run_times)
        print("N-Queens - MIMIC avg run time,", hp_value, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "N-Queens MIMIC - " \
        + str(population_size) + " pop, " \
        + ": fit vs iter"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/n_queens_mimic_fitness.png')
    plt.clf()

    return fitness_dfs


def main():
    verbose = True
    num_runs = 20
    max_iters = 500

    # define custom fitness function to maximize instead of minimize
    fitness_fn = mlrose.CustomFitness(queens_max)

    # define optimization problem
    length = 16
    nq_problem = mlrose.QueensOpt(
        length=length,
        fitness_fn=fitness_fn,
        maximize=True,
    )

    nq_problem.set_mimic_fast_mode(True)

    # set initial state
    initial_state = np.random.randint(0, length, size=length)

    # randomized hill climbing
    rhc_fitness_dfs = n_queens_rhc(nq_problem, initial_state, max_iters, num_runs, verbose)
    print('---')

    # simulated annealing
    sa_fitness_dfs = n_queens_sa(nq_problem, initial_state, max_iters, num_runs, verbose)
    print('---')

    # genetic algorithm
    ga_fitness_dfs = n_queens_ga(nq_problem, max_iters, num_runs, verbose)
    print('---')

    # MIMIC algorithm
    mimic_fitness_dfs = n_queens_mimic(nq_problem, max_iters, num_runs, verbose)
    print('---')

    # compare algorithm performance
    plotting.compare_algos(
        problem_name='n_queens',
        rhc_dfs=rhc_fitness_dfs,
        sa_dfs=sa_fitness_dfs,
        ga_dfs=ga_fitness_dfs,
        mimic_dfs=mimic_fitness_dfs,
    )


if __name__ == "__main__":
    main()
