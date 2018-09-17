import numpy as np
import math
def euclid_distance(X1, X2):
    #return np.linalg.norm(X1, X2)
    sqrsum = 0
    for i in range(len(X1)):
        sqrsum += (X1[i] - X2[i]) ** 2
    return math.sqrt(sqrsum)

class k_means():
    """K defines the number of clusters that the algorithm needs to find.
       X is the training X, and y is the training y,
       num_iters is the number of iterations that needs to perform,
       is_noisy is a boolean to control the stdout"""
    def __init__(self, k, X, y, num_iters, is_noisy, tolerance):
        self.is_noisy = is_noisy
        self.random_initialize(k, X)
        self.num_iters = num_iters
        self.clusters = [ []  for i in range(k)]
        self.X = X
        self.tolerance = tolerance
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
            self.clusters = [[] for i in range(0, len(self.centroids))]
            for x in self.X:
                distances = [ euclid_distance(x, centroid) for centroid in self.centroids ]
                self.clusters[distances.index(min(distances))].append(x)
            old_centroids = self.centroids
            self.update_centroids()
            if(self.is_noisy):
                print("> epoch {0} completed, new centroids: {1}".format(i, self.centroids_toString()))
            if(self.tolerance!=0):
                if(self.is_noisy):
                    print("Checking tolerance ...")
                if(self.check_tolerance(old_centroids)):
                    break

    def centroids_toString(self):
        string_centroids = []
        for centroid in self.centroids:
            string_centroids.append(np.array2string(centroid, precision=2, separator=',', suppress_small=True))
        return ','.join(string_centroids)

    def train(self):
        self.classify()
        resultX = []
        resultLabel = []
        for i in range(0,len(self.clusters)):
            for item in self.clusters[i]:
                resultX.append(item)
                resultLabel.append(i)
        return (resultX, resultLabel)


    def update_centroids(self):
        for i in range(len(self.clusters)):
            cur_cluster = self.clusters[i]
            self.centroids[i] = np.average(cur_cluster, axis = 0)

    def check_tolerance(self, old_centroids):
        isOptimal = False
        for i in range(len(self.centroids)):
            isOptimal = isOptimal and np.sum((old_centroids[i] - self.centroids[i]) / self.centroids[i] * 100) < tolerance
        return isOptimal
