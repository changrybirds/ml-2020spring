import numpy as np
import pandas as pd

from hiive.mdptoolbox import mdp

import frozen_lake as frz
import plotting

GRID_DIM = 20
PROB_FROZEN = 0.8

DIRECTIONS = {
    0: '←',
    1: '↓',
    2: '→',
    3: '↑',
}


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
    p = np.array(policy).reshape(GRID_DIM, GRID_DIM)
    print("    " + " ".join("%2d" % f for f in range(GRID_DIM)))
    print("    " + "---" * GRID_DIM)
    for x in range(GRID_DIM):
        print(" %2d| " % x + " ".join("%s " % p[x, f] for f in
                                     range(GRID_DIM)))
    print()


def calc_reward(optimal_policy, R):
    """
    Arguments
    ---------
    optimal_policy : tuple
        optimal policy for each state
    R : tuple of ndarrays
        reward matrices, one for each state
    """

    rewards = np.zeros(len(optimal_policy))
    for p in range(len(optimal_policy)):
        policy = optimal_policy[p]

        # get reward for optimal policy
        policy_reward = R[p][policy]
        rewards[p] = policy_reward

    return rewards


# convert from gym to mdpt format - adapted from Blake Wang's suggestion:
# https://piazza.com/class/k51r1vdohil5g3?cid=709
def gym_to_mdpt(env):
    nA = env.nA
    nS = env.nS
    P = np.zeros([nA, nS, nS])
    R = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for p_trans, next_s, reward, _ in transitions:
                P[a, s, next_s] += p_trans
                R[s, a] = reward
            P[a, s, :] /= np.sum(P[a, s, :])

    env.render()
    print()

    return P, R


def create_fl_env(size=GRID_DIM, p=PROB_FROZEN, h_reward=-1.0, step_penalty=-0.05, is_slippery=True):
    lake_grid = frz.generate_random_map(size=GRID_DIM, p=PROB_FROZEN)
    fl_env = frz.FrozenLakeEnv(
        desc=lake_grid,
        h_reward=-h_reward,
        step_penalty=step_penalty,
        is_slippery=is_slippery,
    )

    return fl_env


def output_to_csv(filename, rewards, iters):
    d = {'rewards': rewards, 'iters': iters}
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
        P, R = gym_to_mdpt(env)
        fl_vi = mdp.ValueIteration(
            transitions=P,
            reward=R,
            gamma=gamma,
            max_iter=max_iters,
        )
        # if verbose: fl_vi.setVerbose()
        fl_vi.run()

        # add error means for each episode
        error_m = np.sum(fl_vi.error_mean)
        all_error_means.append(error_m)
        print("Frozen Lake VI Episode", episode, "error mean:", error_m, '\n')

        error_over_iters = fl_vi.error_over_iters
        # print(error_over_iters)
        error_plot_df = pd.DataFrame(0, index=np.arange(1, max_iters + 1), columns=['error'])
        error_plot_df.iloc[0:len(error_over_iters), :] = error_over_iters
        all_error_dfs.append(error_plot_df)

        policy_arrows = [DIRECTIONS[direction] for direction in fl_vi.policy]
        print_policy(policy_arrows)

        rewards = calc_reward(fl_vi.policy, R)
        total_reward = np.sum(rewards)
        all_rewards.append(total_reward)
        print("Frozen Lake VI Episode", episode, "reward:", total_reward, '\n')

        all_iters.append(fl_vi.iter)
        print("Frozen Lake VI Episode", episode, "last iter:", fl_vi.iter, '\n')

    filename = "fl_vi_stats.csv"
    output_to_csv(filename, all_rewards, all_iters)

    combined_error_df = pd.concat(all_error_dfs, axis=1)
    mean_error_per_iter = combined_error_df.mean(axis=1)
    mean_error_per_iter.to_csv("tmp/fl_vi_error.csv")

    # plot the error over iterations
    title = "FL VI: error vs. iter (mean over " + str(num_episodes) + " episodes)"
    path = "graphs/fl_vi_error_iter.png"
    plotting.plot_error_over_iters(mean_error_per_iter, title, path, xlim=50)


def run_pi(envs, gamma=0.96, max_iters=1000, verbose=True):
    all_rewards = []
    all_iters = []
    all_error_means = []
    all_error_dfs = []

    num_episodes = len(envs)
    for env, episode in zip(envs, range(num_episodes)):
        P, R = gym_to_mdpt(env)
        fl_pi = mdp.PolicyIteration(
            transitions=P,
            reward=R,
            gamma=gamma,
            max_iter=max_iters,
        )
        # if verbose: fl_pi.setVerbose()
        fl_pi.run()

        # add error means for each episode
        error_m = np.sum(fl_pi.error_mean)
        all_error_means.append(error_m)
        print("Frozen Lake PI Episode", episode, "error mean:", error_m, '\n')

        error_over_iters = fl_pi.error_over_iters
        variation_over_iters = fl_pi.variation_over_iters
        # print(error_over_iters)
        error_plot_df = pd.DataFrame(0, index=np.arange(1, max_iters + 1), columns=['error'])
        error_plot_df.iloc[0:len(error_over_iters), :] = error_over_iters
        all_error_dfs.append(error_plot_df)

        policy_arrows = [DIRECTIONS[direction] for direction in fl_pi.policy]
        print_policy(policy_arrows)

        rewards = calc_reward(fl_pi.policy, R)
        total_reward = np.sum(rewards)
        all_rewards.append(total_reward)
        print("Frozen Lake PI Episode", episode, "reward:", total_reward, '\n')

        all_iters.append(fl_pi.iter)
        print("Frozen Lake PI Episode", episode, "last iter:", fl_pi.iter, '\n')

    filename = "fl_pi_stats.csv"
    output_to_csv(filename, all_rewards, all_iters)

    combined_error_df = pd.concat(all_error_dfs, axis=1)
    mean_error_per_iter = combined_error_df.mean(axis=1)
    mean_error_per_iter.to_csv("tmp/fl_pi_error.csv")

    # plot the error over iterations
    title = "FL PI: error vs. iter (mean over " + str(num_episodes) + " episodes)"
    path = "graphs/fl_pi_error_iter.png"
    plotting.plot_error_over_iters(mean_error_per_iter, title, path, xlim=50)


def run_qlearn(envs, gamma=0.96, n_iters=10000, verbose=True):
    all_rewards = []
    all_mean_discrepancies_dfs = []
    all_error_dfs = []

    num_episodes = len(envs)
    for env, episode in zip(envs, range(num_episodes)):
        P, R = gym_to_mdpt(env)
        fl_qlearn = mdp.QLearning(
            transitions=P,
            reward=R,
            gamma=gamma,
            n_iter=n_iters,
        )
        # if verbose: fl_qlearn.setVerbose()
        fl_qlearn.run()

        # add mean discrepancies for each episode
        v_means = []
        for v_mean in fl_qlearn.v_mean:
            v_means.append(np.mean(v_mean))
        v_mean_df = pd.DataFrame(v_means, columns=['v_mean'])
        # v_mean_df.iloc[0: n_iters / 100, :] = v_means

        all_mean_discrepancies_dfs.append(v_mean_df)
        print("Frozen Lake QLearning Episode", episode, "mean discrepancy:", '\n', v_mean_df, '\n')

        error_over_iters = fl_qlearn.error_over_iters
        # print(error_over_iters)
        error_plot_df = pd.DataFrame(0, index=np.arange(1, n_iters + 1), columns=['error'])
        error_plot_df.iloc[0:len(error_over_iters), :] = error_over_iters
        all_error_dfs.append(error_plot_df)

        policy_arrows = [DIRECTIONS[direction] for direction in fl_qlearn.policy]
        print_policy(policy_arrows)

        rewards = calc_reward(fl_qlearn.policy, R)
        total_reward = np.sum(rewards)
        all_rewards.append(total_reward)
        print("Frozen Lake QLearning Episode", episode, "reward:", total_reward, '\n')

    filename = "tmp/fl_qlearn_stats.csv"
    rewards_df = pd.DataFrame(all_rewards)
    rewards_df.to_csv(filename)

    mean_v_filename = "tmp/fl_qlearn_meanv.csv"
    mean_v_df = pd.concat(all_mean_discrepancies_dfs, axis=1)
    mean_mean_v = mean_v_df.mean(axis=1)
    mean_mean_v.to_csv(mean_v_filename)

    combined_error_df = pd.concat(all_error_dfs, axis=1)
    mean_error_per_iter = combined_error_df.mean(axis=1)
    mean_error_per_iter.to_csv("tmp/fl_qlearn_error.csv")

    # plot the error over iterations
    title = "FL QL: error vs. iter (mean over " + str(num_episodes) + " episodes)"
    path = "graphs/fl_ql_error_iter.png"
    plotting.plot_error_over_iters(mean_error_per_iter, title, path)


def main():
    verbose = True
    max_iters = 1000
    num_episodes = 100
    gamma = 0.96

    n_iters = 10000  # for Q learning

    fl_envs = []
    for e in range(num_episodes):
        fl_env = create_fl_env(step_penalty=-0.05)
        fl_envs.append(fl_env)

    run_vi(fl_envs, gamma=gamma, max_iters=max_iters, verbose=verbose)
    run_pi(fl_envs, gamma=gamma, max_iters=max_iters, verbose=verbose)
    run_qlearn(fl_envs, gamma=gamma, n_iters=n_iters, verbose=verbose)


if __name__ == "__main__":
    main()
