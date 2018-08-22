import cPickle, gzip, numpy

#load mnist dataset from my own mnist directory
f = gzip.open('./../../../dataset/mnist/mnist.pl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
