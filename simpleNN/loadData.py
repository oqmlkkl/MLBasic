import cPickle, gzip, numpy

#load mnist dataset from my own mnist directory
f = gzip.open('./../../../dataset/mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
# print(train_set[0][0].shape)
f.close()
