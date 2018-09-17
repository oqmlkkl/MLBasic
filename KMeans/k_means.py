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
        self.num_iters = num_iters
        # Convert each X into a tuple where the second element is the class
        self.X = [(x, 0) for x in X]
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

    def classify(self):
        for i in range(self.num_iters):
            for x in self.X:
                distances = [ euclid_distance(x[0], centroid) in self.centroids ]
                x[1] = distances.index(min(distances))
            self.update_centroids()
            if(self.is_noisy):
                print "> epoch {0} completed, new centroids: {1}".format(i, self.centroids)

    def udpate_centroids(self):
        #TODO

