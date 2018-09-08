from feedforwardNN import feedforwardNN
from loadData import data_set
import pdb

#define parameters
data_set = data_set()
num_hidden_layer = 1
neurons = 30
learning_rate = 0.05
batch_size = 30
epoch = 50

#start training
network = feedforwardNN(num_hidden_layer, neurons, data_set.get_trainset(), learning_rate)
network.stochastic_gradient_descent(epoch, batch_size, data_set.get_testset())
