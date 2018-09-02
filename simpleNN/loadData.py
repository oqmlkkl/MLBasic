import cPickle, gzip
import numpy as np

class data_set:

    def __init__(self):
        #load mnist dataset from my own mnist directory
        f = gzip.open('./../../../dataset/mnist/mnist.pkl.gz', 'rb')
        self.train_set, self.valid_set, self.test_set = cPickle.load(f)
        f.close()
        #print(train_set[0].shape)

    def get_trainset(self):
        train_x = self.train_set[0]
        train_y = self.train_set[1]
        #print(len(train_y))
        # vectorize y
        train_y_vector = np.zeros((len(train_y), 10))
        for i in range(0, len(train_y)):
            train_y_vector[i][train_y[i]] = 1
        #print(train_y_vector)
        return train_x, train_y_vector


