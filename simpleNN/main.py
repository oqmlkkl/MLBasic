from feedforwardNN import feedforwardNN
from loadData import data_set
import pdb
data_set = data_set()
X, y = data_set.get_trainset()
neurons = 10
network = feedforwardNN(10, neurons, 10, X, y, 0.05, X.shape[0])
for i in range(X.shape[0]):
    network.train(X[i], y[i])
