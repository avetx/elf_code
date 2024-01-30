import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

from elf import elf_log_regression
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def estimate_remaining_time(iteration, total_iterations, start_time):
    elapsed_time = time.time() - start_time
    iterations_left = total_iterations - iteration
    time_per_iteration = elapsed_time / iteration if iteration > 0 else 0
    remaining_time = iterations_left * time_per_iteration
    return remaining_time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load a9a dataset
# dataset = 'a9a'

# X, y = load_svmlight_file(f'{dataset}.txt')
# # Convert to dense numpy array
# X = X.toarray()
# # Convert labels to 0 (edible) and 1 (poisonous)
# y = (y == 1).astype(int)

# fetch glioma dataset
# glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
#
# # data (as pandas dataframes)
# X = glioma_grading_clinical_and_mutation_features.data.features
# y = glioma_grading_clinical_and_mutation_features.data.targets



lamb = 1


#choosing the method
method_list = ['B-ELF', 'P-ELF', 'D-ELF', 'LMC']

#choosing the dataset
dataset_list = ['a8a', 'a9a', 'mushrooms']

#We use the compressor Top-tau
tau_list = [5, 10, 50]

#Stepsize list
gamma_list = [0.01, 0.1, 0.5, 1]


# The frequency that we save the iterations of ELF with.
freq = 10
param_list = [ ['a8a', 1, 5], ['a8a', 1, 10],['a8a', 1, 5],['a8a', 1, 50],  ['a9a', 1, 5], ['a9a', 1, 10],['a9a', 1, 50], ['mushrooms', 1, 5], ['mushrooms', 1, 10],['mushrooms', 1, 50]]
# param_list = [ ['a8a', 1, 5]]

for dataset,gamma,tau in param_list:

    print(f'{dataset}, Top-{tau}')
    X, y = load_svmlight_file(f'{dataset}.txt')
    # Convert to dense numpy array
    X = X.toarray()
    # Convert labels to 0 (edible) and 1 (poisonous)
    y = (y == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set parameters
    T = 100000  # Time horizon in terms of communication complexity
    # We average the last m iterates of the resulting algorithm instead of sampling m different iterations.
    m = 100

    # Dividing the data among n clients
    num_client = 40

    n, d = X_train.shape
    size_client = int(n / num_client)
    X_dist = [X_train[i * size_client: (i + 1) * size_client] for i in range(num_client)]
    y_dist = [y_train[i * size_client: (i + 1) * size_client] for i in range(num_client)]

    X_train, y_train = X_train[:num_client * size_client], y_train[:num_client * size_client]

    n, d = X_train.shape


    # Initialize tqdm for progress bar
    #  progress_bar = tqdm(total=len(tau_list), desc=f' dataset = {dataset}, Gamma = {gamma}')

    # The number of coordinates communicated at each iteration.
    method_vector_size = [2 * tau * (1 + np.log(d).astype(int)), d + tau * (1 + np.log(d).astype(int)), d + tau * (1 + np.log(d).astype(int)), 2 * d]
    # Calculate the number of iterations for each method to have the same X-axis in the plots. MAYBE IT IS BETTER NOT TO HAVE IT.
    k_list = [T // size for size in method_vector_size]
    # Initialize the starting point
    x0 = np.random.choice([-1, 1], size=d)  # Replace 10 with the desired size of your array
    # print('starting point -', x0)
    fig, ax = plt.subplots()
    for j in np.arange(len(method_list)):
        seed_numb = np.random.poisson(lam=100, size=1)
        np.random.seed(seed_numb)
        k = k_list[j]
        # RUN ELF and save the iterates in a txt file.
        elf_iterates = elf_log_regression(X_dist, y_dist, k, gamma, x0, d, lamb, method_list[j], num_client, tau,freq)
        # np.savetxt(f'./iterations/T_{T}_gamma_{gamma}_freq_{freq}_Top-{tau}_data_{dataset}_{method_list[j]}_m_{m}_iterates.txt', elf_iterates)

        lensize = len(elf_iterates)
        elf_accuracy_list = []

        for i in range(lensize):
            # Calculate the index range for the rolling average
            start_index = max(0, i - m + 1)
            end_index = i + 1

            # Calculate the rolling average of the selected entries
            rolling_average_final_estimator = np.mean(elf_iterates[start_index:end_index], axis=0)

            # Use the rolling average as final_estimator
            final_estimator = rolling_average_final_estimator
            # print(final_estimator)
            logits_test = np.dot(X_test, final_estimator)
            # Apply the sigmoid function to get probabilities
            probabilities_test = sigmoid(logits_test)
            # Classify based on a threshold (e.g., 0.5)
            predictions_test = (probabilities_test >= 0.5).astype(int)
            # Evaluate the accuracy on the test set
            accuracy_test = np.mean(predictions_test == y_test)

            # Append the accuracy value to the list
            elf_accuracy_list.append(accuracy_test)

        # Convert the list to a NumPy array for plotting
        elf_accuracy_array = np.array(elf_accuracy_list)

        # Plotting
        plt.rcParams.update({'font.size': 15})
        plt.legend(fontsize = 15)
        scaled_x_values = 32 * method_vector_size[j] * np.arange(np.divide(k,freq).astype(int))
        plt.plot(scaled_x_values,elf_accuracy_array[:np.divide(k,freq).astype(int)],label = method_list[j])

    plt.xlabel('Bits')
    plt.ylabel('Test Accuracy')
    plt.title(f'{dataset}, $\gamma$={gamma}, Top-{tau}')
    plt.legend(fontsize=15)
    vertical_lines_x = [5e4, 10e4, 15e4,20e4,25e4,30e4]  # Replace with your desired x-values
    for x_value in vertical_lines_x:
        ax.axvline(x=x_value, linestyle='--', color='gray', alpha=0.5)

    ax.xaxis.set_major_locator(MultipleLocator(base=5e4))  # Set the x-axis locator
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:1.0e}"))  # Format x-axis labels
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}_gamma_{gamma}_Top_{tau}_freq_{freq}_m_{m}.pdf', dpi=600)
    plt.clf()

#


