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
        # vectorize y
        train_y_vector = np.zeros((len(train_y), 10))
        for i in range(0, len(train_y)):
            train_y_vector[i][train_y[i]] = 1
        tr = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(train_x, train_y_vector)]
        print "{0} {1}".format(tr[0][1], train_y[0])
        print "{0} {1}".format(tr[1][1], train_y[1])
        return tr

    def get_testset(self):
        test_x = self.test_set[0]
        test_y = self.test_set[1]
        te = [(x.reshape(784, 1),y) for x, y in zip(test_x, test_y)]
        return te


