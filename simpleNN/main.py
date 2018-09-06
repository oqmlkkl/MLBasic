from feedforwardNN import feedforwardNN
from loadData import data_set
import pdb
data_set = data_set()
neurons = 30
network = feedforwardNN(1, neurons, data_set.get_trainset(), 0.05)
print(data_set.get_trainset()[0][0].shape)
network.stochastic_gradient_descent(30, 30, data_set.get_testset())
