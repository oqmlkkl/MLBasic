import numpy as np
import random

class feedforwardNN:

    #num_hidden_layers is an int represents the number of hidden layers in the NN
    #neurons is a list of int which indicates the number of neurons in each hidden
    #        layer of the NN
    def __init__(self, num_hidden_layers, neurons):
        try:
            self.num_hidden_layers = num_hidden_layers
            self.neurons = neurons
            if(num_hidden_layers != len(neurons)):
                raise ValueError('Cannot continue: neurons input has different length as num_hidden_layers')
            self.random_init(3)

        except Exception as error:
            print(error)

    #randomly initialize theta based on symmetry breaking between range:[-epsion, epsilon]
    def random_init(self, epsilon):
        self.theta = ()
        for i in range(0, self.num_hidden_layers):
            cur_theta = np.full((1, self.neurons[i] +1), random.uniform(-epsilon, epsilon))
            self.theta += (cur_theta)
        print('theta = ' + self.theta)

    def setData(self, training_set, validation_set, test_set):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def activation(self, X, theta):
        z = X * theta.T
        return np.concatenate(np.ones(z.shape[0]).T, self.sigmoid(z))

    def get_init_delta(self):
        delta = ()
        for i in range(1, self.num_hidden_layers):
            delta += (np.full((1, self.neurons[i]), 0))
        return delta

    # implement the back propagation algorithm
    def back_propagation(self):
        # step1: initialize the delta matrix
        delta_matrix = self.get_init_delta()
        # step2: do the forward propagation



    def cost():
