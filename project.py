import numpy as np
import pandas as pd

# Class definition
class matrix:
    def __init__(self, filename):
        self.array_2d = None
        self.load_from_csv(filename)

    def load_from_csv(self, filename):
        """Load data from a CSV file into the array_2d attribute."""
        self.array_2d = np.genfromtxt(filename, delimiter=',', skip_header=1)

    def standardise(self):
        """Standardise the array_2d."""
        means = np.mean(self.array_2d, axis=0)
        maxs = np.max(self.array_2d, axis=0)
        mins = np.min(self.array_2d, axis=0)
        self.array_2d = (self.array_2d - means) / (maxs - mins)

    def get_distance(self, other_matrix, row_i):
        """Calculate the Euclidean distance between a specific row and all rows in another matrix."""
        distances = np.sqrt(np.sum((self.array_2d[row_i] - other_matrix.array_2d) ** 2, axis=1))
        return distances.reshape(-1, 1)

    def get_weighted_distance(self, other_matrix, weights, row_i):
        """Calculate the Weighted Euclidean distance between a specific row and all rows in another matrix."""
        distances = np.sqrt(np.sum(weights.array_2d * (self.array_2d[row_i] - other_matrix.array_2d) ** 2, axis=1))
        return distances.reshape(-1, 1)

    def get_count_frequency(self):
        """Count the frequency of elements if there's only one column."""
        if self.array_2d.shape[1] != 1:
            return 0
        unique, counts = np.unique(self.array_2d, return_counts=True)
        return dict(zip(unique, counts))

# Functions
def get_initial_weights(m):
    """Generate a weight vector with random values summing to 1."""
    weights = np.random.rand(1, m)
    weights /= np.sum(weights)
    return weights

def get_centroids(data_matrix, S, K):
    """Calculate centroids based on the assigned groups."""
    centroids = np.zeros((K, data_matrix.array_2d.shape[1]))
    for k in range(K):
        centroids[k] = np.mean(data_matrix.array_2d[S.flatten() == (k + 1)], axis=0)
    return centroids

def get_separation_within(data_matrix, centroids, S, K):
    """Calculate separation within clusters."""
    separation = np.zeros((1, data_matrix.array_2d.shape[1]))
    for k in range(K):
        cluster_data = data_matrix.array_2d[S.flatten() == (k + 1)]
        separation += np.sum((cluster_data - centroids[k]) ** 2, axis=0)
    return separation.reshape(1, -1)

def get_separation_between(data_matrix, centroids, S, K):
    """Calculate separation between clusters."""
    separation = np.zeros((1, data_matrix.array_2d.shape[1]))
    Nk = np.bincount(S.flatten().astype(int))
    for k in range(K):
        separation += Nk[k] * (centroids[k] ** 2)
    return separation.reshape(1, -1)

def get_groups(data_matrix, K):
    """Assign groups to data based on clustering."""
    n = data_matrix.array_2d.shape[0]
    S = np.zeros((n, 1))
    weights = get_initial_weights(data_matrix.array_2d.shape[1])
    centroids = data_matrix.array_2d[np.random.choice(n, K, replace=False)]

    while True:
        # Step 7: Assign groups
        for i in range(n):
            distances = np.linalg.norm(data_matrix.array_2d[i] - centroids, axis=1)
            S[i] = np.argmin(distances) + 1

        # Update centroids
        new_centroids = get_centroids(data_matrix, S, K)

        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids

    return S

def get_new_weights(data_matrix, centroids, old_weights, S, K):
    """Update the weights based on new centroids."""
    a = get_separation_within(data_matrix, centroids, S, K)
    b = get_separation_between(data_matrix, centroids, S, K)
    m = data_matrix.array_2d.shape[1]
    new_weights = (old_weights + b / a) / (2 * m)
    return new_weights

def run_test():
    m = matrix('C:/Users/91934/OneDrive/Desktop/project/Data (2).csv')
    for k in range(2, 11):
        for i in range(20):
            S = get_groups(m, k)
            print(str(k) + '=' + str(m.get_count_frequency()))

# Uncomment below to run the test function
run_test()
