import numpy as np
import random

class feedforwardNN:

    #num_hidden_layers is an int represents the number of hidden layers in the NN
    #neurons is a list of int which indicates the number of neurons in each hidden
    #        layer of the NN
    def __init__(self, num_hidden_layers, neurons, num_iters):
        try:
            self.num_hidden_layers = num_hidden_layers
            self.neurons = neurons
            if(num_hidden_layers != len(neurons)):
                raise ValueError('Cannot continue: neurons input has different length as num_hidden_layers')
            self.random_init(3)
            self.num_iters = num_iters

        except Exception as error:
            print(error)

    #randomly initialize theta based on symmetry breaking between range:[-epsion, epsilon]
    def random_init(self, epsilon, X, y):
        self.theta = ()
        for i in range(0, self.num_hidden_layers):
            if i == 0:
                self.theta += np.random.rand(X.shape()[1] + 1, self.neurons[0])
            else if i == self.num_hidden_layers - 1:
                self.theta += np.random.rand(self.neurons[i] + 1, y.shape()[1])
            else:
                self.thata += np.random.rand(self.neurons[i] + 1, self.neurons[i + 1])
        print('theta = ' + self.theta)

    def setData(self, training_set, validation_set, test_set):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def get_activations(self, X, theta):
        z = X * theta.T
        return np.concatenate(np.ones(z.shape[0]).T, self.sigmoid(z))

    def get_init_delta(self, X):
        # init delta for input layer
        delta = (np.random.rand(1, shape(X)[1]))
        # init delta for hidden layers
        for i in range(0, self.num_hidden_layers):
            delta += (np.full((1, self.neurons[i]), 0))
        return delta

    # implement the back propagation algorithm once,
    # it returns the delta matrix for the input training example
    def back_propagation(self, X, y):
        # step1: do the forward propagation to get the delta of output,
        #        and store the activations for back propagation
        activations = ()
        delta_matrix = self.get_init_delta(X)
        for i in range(0, self.num_hidden_layers + 1):
            if i == 0:
                activations += (np.concatenate(np.ones(1), X))
            else:
                activations += (np.ones(self.neurons[i]))
        #forward propagation, update activations
        for i in range(1, self.num_hidden_layers):
            activations[i] = self.get_activation(activations[i - 1], self.theta[i - 1])
        # step2: update delta_matrix
        # update the last delta
        delta_matrix[len(delta_matrix) - 1] = activations[len(activations)  - 1] - y
        # according to the last delta, back-propagate to update activations
        for i in range(2, self.num_hidden_layers):
            layer = self.num_hidden_layers - i
            coeff = self.theta[layer].T * delta_matrix[layer + 1]
            deri = np.dot(activations[layer], (1 - activations[layer]))
            delta_matrix[layer] = np.dot(coeff, deri)
        return delta_matrix

    def train(self, X, y):
        delta_matrix= self.get_init_delta(X)
        # num_examples
        m = X.shape()[1]
        for i in range(0, m):
            delta_matrix += self.back_propagation(X.T[i].T, y[i])

