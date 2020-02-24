import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

import mlrose_hiive as mlrose

import plotting


def knapsack_rhc(knapsack_problem, initial_state, max_iters=np.inf, num_runs=20, verbose=False):
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
                problem=knapsack_problem,
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
        print("Knapsack - RHC avg run time,", hp_value, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "Knapsack RHC: fitness vs. iterations"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/knapsack_rhc_fitness.png')
    plt.clf()

    return fitness_dfs


def knapsack_sa(knapsack_problem, initial_state, max_iters=np.inf, num_runs=20, verbose=False):
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
                problem=knapsack_problem,
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
        print("Knapsack - SA avg run time,", hp_value_string, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "Knapsack SA: fitness vs. iterations"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values_strings,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/knapsack_sa_fitness.png')
    plt.clf()

    return fitness_dfs


def knapsack_ga(knapsack_problem, max_iters=np.inf, num_runs=20, verbose=False):
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
                problem=knapsack_problem,
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
        print("Knapsack - GA avg run time,", hp_value, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "Knapsack GA - " \
        + str(population_size) + " pop, " \
        + str(mutation_prob) + " mut prob, " \
        + ": fit vs iter"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/knapsack_ga_fitness.png')
    plt.clf()

    return fitness_dfs


def knapsack_mimic(knapsack_problem, max_iters=np.inf, num_runs=20, verbose=False):
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
                problem=knapsack_problem,
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
        print("Knapsack - MIMIC avg run time,", hp_value, hp_name, ":", avg_run_time)

    # generate plots
    plot_title = "Knapsack MIMIC - " \
        + str(population_size) + " pop, " \
        + ": fit vs iter"
    plotting.plot_fitness_curves(
        fitness_dfs=fitness_dfs,
        hp_values=hp_values,
        hp_name=hp_name,
        title=plot_title,
    )
    plt.savefig('graphs/knapsack_mimic_fitness.png')
    plt.clf()

    return fitness_dfs


def main():
    verbose = True
    num_runs = 20
    max_iters = 1000

    # define Knapsack fitness function to maximize
    num_items = 50
    weights = np.random.randint(1, 50, size=num_items)
    values = np.random.randint(1, 10, size=num_items)
    max_weight_pct = 0.5
    fitness_fn = mlrose.Knapsack(
        weights,
        values,
        max_weight_pct=max_weight_pct,
    )

    # define optimization problem
    knapsack_problem = mlrose.KnapsackOpt(
        fitness_fn=fitness_fn,
    )

    knapsack_problem.set_mimic_fast_mode(True)

    # set initial state
    count = 0
    while True:
        if verbose: print(count)
        initial_state = np.random.randint(0, 3, size=num_items)
        total_weight = np.sum(np.multiply(initial_state, weights))
        if total_weight <= np.sum(weights) * max_weight_pct:
            break
        count += 1

    if verbose: print(initial_state)

    # randomized hill climbing
    rhc_fitness_dfs = knapsack_rhc(knapsack_problem, initial_state, max_iters, num_runs, verbose)
    print('---')

    # simulated annealing
    sa_fitness_dfs = knapsack_sa(knapsack_problem, initial_state, max_iters, num_runs, verbose)
    print('---')

    # genetic algorithm
    ga_fitness_dfs = knapsack_ga(knapsack_problem, max_iters, num_runs, verbose)
    print('---')

    # MIMIC algorithm
    mimic_fitness_dfs = knapsack_mimic(knapsack_problem, max_iters, num_runs, verbose)
    print('---')

    # compare algorithm performance
    plotting.compare_algos(
        problem_name='knapsack',
        rhc_dfs=rhc_fitness_dfs,
        sa_dfs=sa_fitness_dfs,
        ga_dfs=ga_fitness_dfs,
        mimic_dfs=mimic_fitness_dfs,
    )


if __name__ == "__main__":
    main()
