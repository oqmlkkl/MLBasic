import cPickle, gzip
import numpy as np

#load mnist dataset from my own mnist directory
f = gzip.open('./../../../dataset/mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
#print(train_set[0].shape)

train_x = train_set[0]
train_y = train_set[1]
#print(len(train_y))
# vectorize y
train_y_vector = np.zeros((len(train_y), 10))
for i in range(0, len(train_y)):
    train_y_vector[i][train_y[i]] = 1
#print(train_y_vector)
f.close()
