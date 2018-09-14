import numpy as np

def euclid_distance(X1, X2):
    return np.linalg.norm(X1, X2)

class k_means():
    """K defines the number of clusters that the algorithm needs to find.
       X is the training X, and y is the training y,
       num_iters is the number of iterations that needs to perform,
       is_noisy is a boolean to control the stdout"""
    def __init__(self, k, X, y, num_iters, is_noisy):
        self.is_noisy = is_noisy
        self.random_initialize(k, X)
        if(is_noisy):
            print('Init completed.')

    def random_initialize(self, k, X):
        #do the random_initialization to randomly select centroids
        rand_indexes = np.arange(k * 2)
        np.random.shuffle(rand_indexes)
        rand_indexes = rand_indexes[:k]
        self.centroids = [ X[i] for i in rand_indexes ]
        if(self.is_noisy):
            print('Randomly Selected Centroids: ')
            print(self.centroids)
