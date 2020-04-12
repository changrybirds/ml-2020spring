import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hiive.mdptoolbox import mdp

import plotting


# The number of population abundance classes
POPULATION_CLASSES = 8
# The number of years since a fire classes
FIRE_CLASSES = 14
# The number of states
STATES = POPULATION_CLASSES * FIRE_CLASSES
# The number of actions
ACTIONS = 2
ACTION_NOTHING = 0
ACTION_BURN = 1
# Probability a population remains in its current abundance class
PROB_REMAIN = 0.5


def check_action(x):
    """Check that the action is in the valid range."""
    if not (0 <= x < ACTIONS):
        msg = "Invalid action '%s', it should be in {0, 1}." % str(x)
        raise ValueError(msg)


def check_population_class(x):
    """Check that the population abundance class is in the valid range."""
    if not (0 <= x < POPULATION_CLASSES):
        msg = "Invalid population class '%s', it should be in {0, 1, …, %d}." \
              % (str(x), POPULATION_CLASSES - 1)
        raise ValueError(msg)


def check_fire_class(x):
    """Check that the time in years since last fire is in the valid range."""
    if not (0 <= x < FIRE_CLASSES):
        msg = "Invalid fire class '%s', it should be in {0, 1, …, %d}." % \
              (str(x), FIRE_CLASSES - 1)
        raise ValueError(msg)


def check_probability(x, name="probability"):
    """Check that a probability is between 0 and 1."""
    if not (0 <= x <= 1):
        msg = "Invalid %s '%s', it must be in [0, 1]." % (name, str(x))
        raise ValueError(msg)


def get_habitat_suitability(years):
    """The habitat suitability of a patch relatve to the time since last fire.

    The habitat quality is low immediately after a fire, rises rapidly until
    five years after a fire, and declines once the habitat is mature. See
    Figure 2 in Possingham and Tuck (1997) for more details.

    Parameters
    ----------
    years : int
        The time in years since last fire.

    Returns
    -------
    r : float
        The habitat suitability.

    """
    if years < 0:
        msg = "Invalid years '%s', it should be positive." % str(years)
        raise ValueError(msg)
    if years <= 5:
        return 0.2*years
    elif 5 <= years <= 10:
        return -0.1*years + 1.5
    else:
        return 0.5


def convert_state_to_index(population, fire):
    """Convert state parameters to transition probability matrix index.

    Parameters
    ----------
    population : int
        The population abundance class of the threatened species.
    fire : int
        The time in years since last fire.

    Returns
    -------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.

    """
    check_population_class(population)
    check_fire_class(fire)
    return population*FIRE_CLASSES + fire


def convert_index_to_state(index):
    """Convert transition probability matrix index to state parameters.

    Parameters
    ----------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.

    Returns
    -------
    population, fire : tuple of int
        ``population``, the population abundance class of the threatened
        species. ``fire``, the time in years since last fire.

    """
    if not (0 <= index < STATES):
        msg = "Invalid index '%s', it should be in {0, 1, …, %d}." % \
              (str(index), STATES - 1)
        raise ValueError(msg)
    population = index // FIRE_CLASSES
    fire = index % FIRE_CLASSES
    return (population, fire)


def transition_fire_state(F, a):
    """Transition the years since last fire based on the action taken.

    Parameters
    ----------
    F : int
        The time in years since last fire.
    a : int
        The action undertaken.

    Returns
    -------
    F : int
        The time in years since last fire.

    """
    ## Efect of action on time in years since fire.
    if a == ACTION_NOTHING:
        # Increase the time since the patch has been burned by one year.
        # The years since fire in patch is absorbed into the last class
        if F < FIRE_CLASSES - 1:
            F += 1
    elif a == ACTION_BURN:
        # When the patch is burned set the years since fire to 0.
        F = 0

    return F


def get_transition_probabilities(s, x, F, a):
    """Calculate the transition probabilities for the given state and action.

    Parameters
    ----------
    s : float
        The class-independent probability of the population staying in its
        current population abundance class.
    x : int
        The population abundance class of the threatened species.
    F : int
        The time in years since last fire.
    a : int
        The action undertaken.

    Returns
    -------
    prob : array
        The transition probabilities as a vector from state (``x``, ``F``) to
        every other state given that action ``a`` is taken.

    """
    # Check that input is in range
    check_probability(s)
    check_population_class(x)
    check_fire_class(F)
    check_action(a)

    # a vector to store the transition probabilities
    prob = np.zeros(STATES)

    # the habitat suitability value
    r = get_habitat_suitability(F)
    F = transition_fire_state(F, a)

    ## Population transitions
    if x == 0:
        # population abundance class stays at 0 (extinct)
        new_state = convert_state_to_index(0, F)
        prob[new_state] = 1
    elif x == POPULATION_CLASSES - 1:
        # Population abundance class either stays at maximum or transitions
        # down
        transition_same = x
        transition_down = x - 1
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == ACTION_BURN:
            transition_same -= 1
            transition_down -= 1
        # transition probability that abundance stays the same
        new_state = convert_state_to_index(transition_same, F)
        prob[new_state] = 1 - (1 - s)*(1 - r)
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        prob[new_state] = (1 - s)*(1 - r)
    else:
        # Population abundance class can stay the same, transition up, or
        # transition down.
        transition_same = x
        transition_up = x + 1
        transition_down = x - 1
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == ACTION_BURN:
            transition_same -= 1
            transition_up -= 1
            # Ensure that the abundance class doesn't go to -1
            if transition_down > 0:
                transition_down -= 1
        # transition probability that abundance stays the same
        new_state = convert_state_to_index(transition_same, F)
        prob[new_state] = s
        # transition probability that abundance goes up
        new_state = convert_state_to_index(transition_up, F)
        prob[new_state] = (1 - s)*r
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        # In the case when transition_down = 0 before the effect of an action
        # is applied, then the final state is going to be the same as that for
        # transition_same, so we need to add the probabilities together.
        prob[new_state] += (1 - s)*(1 - r)

    # Make sure that the probabilities sum to one
    assert (prob.sum() - 1) < np.spacing(1)
    return prob


def get_transition_and_reward_arrays(s):
    """Generate the fire management transition and reward matrices.

    The output arrays from this function are valid input to the mdptoolbox.mdp
    classes.

    Let ``S`` = number of states, and ``A`` = number of actions.

    Parameters
    ----------
    s : float
        The class-independent probability of the population staying in its
        current population abundance class.

    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrices P and
        ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
        numpy array and R is a numpy vector of length ``S``.

    """
    check_probability(s)

    # The transition probability array
    transition = np.zeros((ACTIONS, STATES, STATES))
    # The reward vector
    reward = np.zeros(STATES)
    # Loop over all states
    for idx in range(STATES):
        # Get the state index as inputs to our functions
        x, F = convert_index_to_state(idx)
        # The reward for being in this state is 1 if the population is extant
        if x != 0:
            reward[idx] = 1
        # Loop over all actions
        for a in range(ACTIONS):
            # Assign the transition probabilities for this state, action pair
            transition[a][idx] = get_transition_probabilities(s, x, F, a)

    return (transition, reward)


def print_policy(policy):
    """Print out a policy vector as a table to console

    Let ``S`` = number of states.

    The output is a table that has the population class as rows, and the years
    since a fire as the columns. The items in the table are the optimal action
    for that population class and years since fire combination.

    Parameters
    ----------
    p : array
        ``p`` is a numpy array of length ``S``.

    """
    p = np.array(policy).reshape(POPULATION_CLASSES, FIRE_CLASSES)
    print("    " + " ".join("%2d" % f for f in range(FIRE_CLASSES)))
    print("    " + "---" * FIRE_CLASSES)
    for x in range(POPULATION_CLASSES):
        print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in
                                     range(FIRE_CLASSES)))

# ---------------------------------------------------------------------------------------- #
# ideally, the above code would be refactored into a class specifically designed to create
# forest manageemnt problems - I'm leaving it as is here due to time constraints


def output_to_csv(filename, iters, rewards=None):
    if rewards is not None and len(rewards) > 0:
        d = {'rewards': rewards, 'iters': iters}
    else:
        d = {'iters': iters}
    df = pd.DataFrame(d)
    path = "tmp/"+ filename
    df.to_csv(path)


def run_vi(envs, gamma=0.96, max_iters=1000, verbose=True):
    all_rewards = []
    all_iters = []
    all_error_means = []
    all_error_dfs = []

    num_episodes = len(envs)
    for env, episode in zip(envs, range(num_episodes)):
        P, R = env
        fm_vi = mdp.ValueIteration(
            transitions=P,
            reward=R,
            gamma=gamma,
            max_iter=max_iters,
        )
        # if verbose: fm_vi.setVerbose()
        fm_vi.run()

        # add error means for each episode
        error_m = np.sum(fm_vi.error_mean)
        all_error_means.append(error_m)
        print("Forest Management VI Episode", episode, "error mean:", error_m, '\n')

        error_over_iters = fm_vi.error_over_iters
        # print(error_over_iters)
        error_plot_df = pd.DataFrame(0, index=np.arange(1, max_iters + 1), columns=['error'])
        error_plot_df.iloc[0:len(error_over_iters), :] = error_over_iters
        all_error_dfs.append(error_plot_df)

        print_policy(fm_vi.policy)
        # print(fm_vi.policy, '\n', R, '\n')

        # rewards = calc_reward(fm_vi.policy, R)
        # total_reward = np.sum(rewards)
        # all_rewards.append(total_reward)
        # print("Forest Management VI Episode", episode, "reward:", total_reward, '\n')

        all_iters.append(fm_vi.iter)
        print("Forest Management VI Episode", episode, "last iter:", fm_vi.iter, '\n')

    filename = "fm_vi_stats.csv"
    output_to_csv(filename, all_iters, all_rewards)

    combined_error_df = pd.concat(all_error_dfs, axis=1)
    mean_error_per_iter = combined_error_df.mean(axis=1)
    mean_error_per_iter.to_csv("tmp/fm_vi_error.csv")

    # plot the error over iterations
    title = "FM VI: error vs. iter (mean over " + str(num_episodes) + " episodes)"
    path = "graphs/fm_vi_error_iter.png"
    plotting.plot_error_over_iters(mean_error_per_iter, title, path, xlim=200)


def run_pi(envs, gamma=0.96, max_iters=1000, verbose=True):
    all_rewards = []
    all_iters = []
    all_error_means = []
    all_error_dfs = []

    num_episodes = len(envs)
    for env, episode in zip(envs, range(num_episodes)):
        P, R = env
        fm_pi = mdp.PolicyIteration(
            transitions=P,
            reward=R,
            gamma=0.96,
        )
        # if verbose: fm_pi.setVerbose()
        fm_pi.run()

        # add error means for each episode
        error_m = np.sum(fm_pi.error_mean)
        all_error_means.append(error_m)
        print("Forest Management PI Episode", episode, "error mean:", error_m, '\n')

        error_over_iters = fm_pi.error_over_iters
        variation_over_iters = fm_pi.variation_over_iters
        # print(error_over_iters)
        error_plot_df = pd.DataFrame(0, index=np.arange(1, max_iters + 1), columns=['error'])
        error_plot_df.iloc[0:len(error_over_iters), :] = error_over_iters
        all_error_dfs.append(error_plot_df)

        print_policy(fm_pi.policy)
        # print(fm_pi.policy, '\n', R, '\n')

        # rewards = calc_reward(fm_pi.policy, R)
        # total_reward = np.sum(rewards)
        # all_rewards.append(total_reward)
        # print("Forest Management PI Episode", episode, "reward:", total_reward, '\n')

        all_iters.append(fm_pi.iter)
        print("Forest Management PI Episode", episode, "last iter:", fm_pi.iter, '\n')

    filename = "fm_pi_stats.csv"
    output_to_csv(filename, all_rewards, all_iters)

    combined_error_df = pd.concat(all_error_dfs, axis=1)
    mean_error_per_iter = combined_error_df.mean(axis=1)
    mean_error_per_iter.to_csv("tmp/fm_pi_error.csv")

    # plot the error over iterations
    title = "FM PI: error vs. iter (mean over " + str(num_episodes) + " episodes)"
    path = "graphs/fm_pi_error_iter.png"
    plotting.plot_error_over_iters(mean_error_per_iter, title, path, xlim=200)


def run_qlearn(envs, gamma=0.96, n_iters=10000, verbose=True):
    all_rewards = []
    all_mean_discrepancies_dfs = []
    all_error_dfs = []

    num_episodes = len(envs)
    for env, episode in zip(envs, range(num_episodes)):
        P, R = env
        fm_qlearn = mdp.QLearning(
            transitions=P,
            reward=R,
            gamma=gamma,
            n_iter=n_iters,
        )
        # if verbose: fm_qlearn.setVerbose()
        fm_qlearn.run()

        # add mean discrepancies for each episode
        v_means = []
        for v_mean in fm_qlearn.v_mean:
            v_means.append(np.mean(v_mean))
        v_mean_df = pd.DataFrame(v_means, columns=['v_mean'])
        # v_mean_df.iloc[0: n_iters / 100, :] = v_means

        all_mean_discrepancies_dfs.append(v_mean_df)
        print("Forest Management QLearning Episode", episode, "mean discrepancy:", '\n', v_mean_df, '\n')

        error_over_iters = fm_qlearn.error_over_iters
        # print(error_over_iters)
        error_plot_df = pd.DataFrame(0, index=np.arange(1, n_iters + 1), columns=['error'])
        error_plot_df.iloc[0:len(error_over_iters), :] = error_over_iters
        all_error_dfs.append(error_plot_df)

        print_policy(fm_qlearn.policy)

        # rewards = calc_reward(fm_qlearn.policy, R)
        # total_reward = np.sum(rewards)
        # all_rewards.append(total_reward)
        # print("Forest Management QLearning Episode", episode, "reward:", total_reward, '\n')

    # filename = "tmp/fm_qlearn_stats.csv"
    # rewards_df = pd.DataFrame(all_rewards)
    # rewards_df.to_csv(filename)

    combined_error_df = pd.concat(all_error_dfs, axis=1)
    mean_error_per_iter = combined_error_df.mean(axis=1)
    mean_error_per_iter.to_csv("tmp/fm_qlearn_error.csv")

    # plot the error over iterations
    title = "FM QL: error vs. iter (mean over " + str(num_episodes) + " episodes)"
    path = "graphs/fm_ql_error_iter.png"
    plotting.plot_error_over_iters(mean_error_per_iter, title, path)


def main():
    verbose = True
    max_iters = 1000
    num_episodes = 100
    gamma = 0.96

    n_iters = 10000  # for Q learning

    fm_envs = []
    for e in range(num_episodes):
        fm_env = get_transition_and_reward_arrays(PROB_REMAIN)
        fm_envs.append(fm_env)

    run_vi(fm_envs, gamma=gamma, max_iters=max_iters, verbose=verbose)
    run_pi(fm_envs, gamma=gamma, max_iters=max_iters, verbose=verbose)
    run_qlearn(fm_envs, gamma=gamma, n_iters=n_iters, verbose=verbose)


if __name__ == "__main__":
    main()
