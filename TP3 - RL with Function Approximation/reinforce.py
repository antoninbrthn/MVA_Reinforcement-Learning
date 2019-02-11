import utils
import tqdm
import pandas as pd
import numpy as np
from math import *

def reinforce(env, N, T, n_itr, discount, learning_rate, init_policy, update_fun = "fixed", from_scratch = True, returns = None,
              parameters = None, bonus = False):

    # Initialize the update function
    assert update_fun in ["annealing", "fixed"]
    if update_fun == "annealing":
        update = lambda x, t: x*learning_rate/(t+1)
    else:
        update = lambda x, t: x * learning_rate

    # Initialize the output or use the ones provided
    if not from_scratch:
        assert (returns is not None) & (parameters is not None)
        returns_dt = returns
        parameters_dt = parameters
    else:
        returns_dt = pd.DataFrame(columns=["Learning Rate", "Stepper", "N","Iteration", "Returns"])
        parameters_dt = pd.DataFrame(columns=["Learning Rate", "Stepper", "N","Iteration", "Params"])

    policy = init_policy


    # Main loop
    theta = policy.theta
    for j in tqdm.tqdm(range(n_itr)):
        paths = utils.collect_episodes(env, policy = policy, horizon = T, n_episodes = N, bonus = bonus)

        # REINFORCE estimates
        returns_tab = []
        grad_J_tab = []
        for p in paths:
            grad_J_episode = 0
            discounted_returns = sum([discount ** i * p["rewards"][i] for i in range(T)])
            returns_tab.append(discounted_returns)
            returns_dt = returns_dt.append({'Learning Rate': str("10^{}".format(int(log10(learning_rate)))),
                                            'Stepper': update_fun,
                                            'N': ". {}".format(str(N)),
                                            'Iteration': j,
                                            "Returns": discounted_returns}, ignore_index = True)
            for i in range(T):
                grad_J_episode += policy.grad_log(p["actions"][i], p["states"][i]) * sum(
                    [discount ** j * p["rewards"][j] for j in range(i, T)])
            grad_J_tab.append(grad_J_episode)
            parameters_dt = parameters_dt.append(
                {'Learning Rate': str("10^{}".format(int(log10(learning_rate)))),
                 'Stepper': update_fun,
                 'N': ". {}".format(str(N)),
                 'Iteration': j,
                 "Params": policy.theta + update(grad_J_episode[0], j)}, ignore_index = True)
        grad_J = np.mean(grad_J_tab)

        # Update policy parameter
        theta = theta + update(grad_J, j)
        policy.set_theta(theta)

    return returns_dt, parameters_dt