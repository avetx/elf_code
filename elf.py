from sklearn.datasets import load_svmlight_file
from collections import deque
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler


import concurrent.futures

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Define a function to estimate remaining time
def estimate_remaining_time(iteration, total_iterations, start_time):
    elapsed_time = time.time() - start_time
    iterations_left = total_iterations - iteration
    time_per_iteration = elapsed_time / iteration if iteration > 0 else 0
    remaining_time = iterations_left * time_per_iteration
    return remaining_time

def log_likelihood(X, y, beta):
    logits = np.dot(X, beta)
    log_prob = -np.log1p(np.exp(-y * logits))
    return np.sum(log_prob)
def log_likelihood_gradient(X, y, beta, lamb):
    logits = np.dot(X, beta)
    probabilities = 1 / (1 + np.exp(-logits))
    return np.dot(X.T, y - probabilities) - lamb * beta

def objective_gradient(X_dist, y_dist, beta, lamb, num_client):

    loc_grads = []
    for j in np.arange(num_client):
        loc_grads.append( log_likelihood_gradient(X_dist[j], y_dist[j] , beta, lamb) )
    return np.average(loc_grads, axis = 0)


def top_tau(v, tau):
    # Calculate absolute values of the coordinates
    abs_values = np.abs(v)
    # Sort indices based on absolute values in descending order
    sorted_indices = np.argsort(abs_values)[::-1]
    # Zero out coordinates that are not among the largest
    v_filtered = np.zeros_like(v)
    v_filtered[sorted_indices[:tau]] = v[sorted_indices[:tau]]
    return v_filtered

def identity_map(v, tau):
    return v


# the distributed data X_dist,y_dist.
# iterates - k
# sampling algorithm 'method'
# stepsize gamma,
# starting point x0.
# There are num_client clients.
# Rand-tau is the compressor
# freq is the frequency that we save the iterates with.
def elf_log_regression(X_dist, y_dist, k, gamma, x0, d, lamb, method, num_client,tau,freq):

    # Initialize tqdm for progress bar
    # progress_bar = tqdm(total=k, desc=f'{method} sampling')

    lamb = 0

    # Define the dual/primal compressor according to the method
    if method == 'LMC':
        Q_D = identity_map
        Q_P = identity_map
    if method == 'D-ELF':
        Q_D = top_tau
        Q_P = identity_map

    if method == 'P-ELF':
        Q_D = identity_map
        Q_P = top_tau

    if method == 'B-ELF':
        Q_D = top_tau
        Q_P = top_tau

    x = x0.copy()

    grad_i = np.zeros((num_client, d))
    # Initialize the grad_i (g_i)
    for j in range(num_client):
        grad_i[j] = log_likelihood_gradient(X_dist[j], y_dist[j], x0, lamb)

    grad = np.average(grad_i,0)
    w = x0.copy()

    all_iterates = []
    for iteration in range(k):

        # THE SERVER
        # Updates the iterate
        x = x + (gamma / 2) * grad + np.sqrt(gamma) * np.random.normal(size=d)
        # Updates the auxiliary sequence w_k using the compressed difference
        w = w + Q_P(x - w,tau)

        # The clients


        for j in range(num_client):
            grad_i[j] = grad_i[j] +  Q_D(log_likelihood_gradient(X_dist[j], y_dist[j], w, lamb) - grad_i[j],tau)

        # The server aggregates the gradient information
        grad = grad + np.average(grad_i,0)

        if np.mod(iteration,freq) == 0:
            all_iterates.append(x.copy())
            # print(grad)


        # Update progress bar
        # progress_bar.update(1)
        # remaining_time = progress_bar.format_dict['elapsed'] / (iteration + 1) * (k - iteration - 1)
        # progress_bar.set_postfix(remaining_time=f"{remaining_time:.2f} s")

        # Close the progress bar
    # progress_bar.close()

        # Return the average of the last iterates
    return all_iterates


a = 1