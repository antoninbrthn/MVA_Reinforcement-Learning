import numpy as np
import lqg1d
import matplotlib.pyplot as plt
import utils
import seaborn as sns
from reinforce import reinforce
import pandas as pd
import copy

plt.rcParams.update({'font.size': 15})


#####################################################
# Define the environment and the policy
#####################################################
env = lqg1d.LQG1D(initial_state_type='random')

# Define base policy
base_policy = utils.Policy(0)

#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 50
# Each trajectory will have at most T time steps
T = 100
# Number of policy parameters updates
n_itr = 50
# Set the discount factor for the problem
discount = 0.9

#####################################################
# Effect of the learning rate, constant stepper
#####################################################
# Include or not
if False:

    # Learning rate for the gradient update
    learning_rates = [1e-3, 1e-4, 1e-5]

    #####################################################
    # define the update rule (stepper)
    stepper = 1 # e.g., constant, adam or anything you want

    # Compute theta_star to study the convergence of the algorithm
    theta_star = float(env.computeOptimalK(discount))
    returns_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Returns"])
    parameters_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Params"])

    for learning_rate in learning_rates:
        returns, params = \
            reinforce(env=env, N = N, T = T, n_itr = n_itr, discount = discount, learning_rate = learning_rate,
                          update_fun = "fixed", init_policy = copy.copy(base_policy), from_scratch = True, bonus = False)
        returns_dt = pd.concat([returns_dt, returns])
        parameters_dt = pd.concat([parameters_dt, params])

    plt.figure(figsize=(8, 6))
    sns.lineplot(data = returns_dt, x='Iteration', y='Returns', hue='Learning Rate')
    plt.title("Average returns for different learning rates")
    plt.savefig("Figures/returns_lr{}.png".format(n_itr))
    plt.yscale("symlog")
    plt.show()

    parameters_dt_plot = parameters_dt[parameters_dt['Learning Rate'] != "10^-3"]
    parameters_dt_plot.loc[:,'Params'] = parameters_dt_plot['Params'].apply(lambda x: (x-theta_star)**2)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data = parameters_dt_plot, x='Iteration', y='Params', hue='Learning Rate')
    plt.ylabel("||theta-theta*||^2")
    plt.yscale("log")
    plt.title("Distance to optimal theta for different learning rates")
    plt.savefig("Figures/params_lr{}.png".format(n_itr))
    plt.show()

#####################################################
# Effect of the learning rate, decreasing stepper
#####################################################
# Include or not
if False:
    # Learning rate for the gradient update
    learning_rate = 1e-4
    steppers = ["fixed", "annealing"]


    # Compute theta_star to study the convergence of the algorithm
    theta_star = float(env.computeOptimalK(discount))
    returns_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Returns"])
    parameters_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Params"])

    for stepper in steppers:
        returns, params = \
            reinforce(env=env, N = N, T = T, n_itr = n_itr, discount = discount, learning_rate = learning_rate,
                          update_fun = stepper, init_policy = copy.copy(base_policy), from_scratch = True, bonus = False)
        returns_dt = pd.concat([returns_dt, returns])
        parameters_dt = pd.concat([parameters_dt, params])

    plt.figure(figsize=(8, 6))
    sns.lineplot(data = returns_dt, x='Iteration', y='Returns', hue='Stepper')
    plt.title("Average returns for different update rules")
    plt.savefig("Figures/step_returns_lr{}.png".format(n_itr))
    plt.yscale("symlog")
    plt.show()

    parameters_dt_plot = parameters_dt[parameters_dt['Learning Rate'] != "10^-3"]
    parameters_dt_plot.loc[:,'Params'] = parameters_dt_plot['Params'].apply(lambda x: (x-theta_star)**2)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data = parameters_dt_plot, x='Iteration', y='Params', hue='Stepper')
    plt.ylabel("||theta-theta*||^2")
    plt.yscale("log")
    plt.title("Distance to optimal theta for different update rules")
    plt.savefig("Figures/step_params_lr{}.png".format(n_itr))
    plt.show()

#####################################################
# Using an exploration bonus
#####################################################
# Include or not
if False:

    # Learning rate for the gradient update
    learning_rates = [1e-4, 1e-5]

    #####################################################
    # define the update rule (stepper)
    stepper = 1  # e.g., constant, adam or anything you want

    # Compute theta_star to study the convergence of the algorithm
    theta_star = float(env.computeOptimalK(discount))
    returns_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Returns"])
    parameters_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Params"])

    for learning_rate in learning_rates:
        returns, params = \
            reinforce(env = env, N = N, T = T, n_itr = n_itr, discount = discount, learning_rate = learning_rate,
                      update_fun = "fixed", init_policy = copy.copy(base_policy), from_scratch = True, bonus = True)
        returns_dt = pd.concat([returns_dt, returns])
        parameters_dt = pd.concat([parameters_dt, params])

    plt.figure(figsize = (8, 6))
    sns.lineplot(data = returns_dt, x = 'Iteration', y = 'Returns', hue = 'Learning Rate')
    plt.title("Average returns for different learning rates with bonus")
    plt.yscale("symlog")
    plt.savefig("Figures/bonus_returns_lr{}_b{}.png".format(n_itr, env.beta))
    plt.show()

    parameters_dt_plot = parameters_dt.copy()
    parameters_dt_plot.loc[:, 'Params'] = parameters_dt_plot['Params'].apply(lambda x: (x - theta_star) ** 2)
    plt.figure(figsize = (8, 6))
    sns.lineplot(data = parameters_dt_plot, x = 'Iteration', y = 'Params', hue = 'Learning Rate')
    plt.ylabel("||theta-theta*||^2")
    plt.yscale("log")
    plt.title("Distance to optimal theta for different learning rates with bonus")
    plt.savefig("Figures/bonus_params_lr{}_b{}.png".format(n_itr, env.beta))
    plt.show()

#####################################################
# Effect of the number of trajectories
#####################################################
# Include or not
if False:

    # Learning rate for the gradient update
    learning_rate = 1e-5
    N_tab = [20 , 50 , 100, 200]
    #####################################################
    # define the update rule (stepper)
    stepper = 1  # e.g., constant, adam or anything you want

    # Compute theta_star to study the convergence of the algorithm
    theta_star = float(env.computeOptimalK(discount))
    returns_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Returns"])
    parameters_dt = pd.DataFrame(columns = ["Learning Rate", "Stepper", "N" ,"Iteration", "Params"])

    for N in N_tab:
        returns, params = \
            reinforce(env = env, N = N, T = T, n_itr = n_itr, discount = discount, learning_rate = learning_rate,
                      update_fun = "fixed", init_policy = copy.copy(base_policy), from_scratch = True, bonus = False)
        returns_dt = pd.concat([returns_dt, returns])
        parameters_dt = pd.concat([parameters_dt, params])

    plt.figure(figsize = (8, 6))
    sns.lineplot(data = returns_dt, x = 'Iteration', y = 'Returns', hue = 'N')
    plt.title("Average returns for different number of trajectories")
    plt.yscale("symlog")
    plt.savefig("Figures/bonus_returns_lr{}_N{}.png".format(n_itr, N))
    plt.show()

    parameters_dt_plot = parameters_dt.copy()
    parameters_dt_plot.loc[:, 'Params'] = parameters_dt_plot['Params'].apply(lambda x: (x - theta_star) ** 2)
    plt.figure(figsize = (8, 6))
    sns.lineplot(data = parameters_dt_plot, x = 'Iteration', y = 'Params', hue = 'N')
    plt.ylabel("||theta-theta*||^2")
    plt.yscale("log")
    plt.title("Distance to optimal theta for different number of trajectories")
    plt.savefig("Figures/bonus_params_lr{}_N{}.png".format(n_itr, N))
    plt.show()


