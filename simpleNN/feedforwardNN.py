import numpy as np
import random
import traceback
import sys

# This class provides the cost function and the function to calculate the delta
class feedforwardNN:

    #num_hidden_layers is an int represents the number of hidden layers in the NN
    #neurons is a list of int which indicates the number of neurons in each hidden
    #        layer of the NN
    def __init__(self, num_hidden_layers, neurons, num_iters, X, y):
        exe_info = sys.exc_info()
        self.num_hidden_layers = num_hidden_layers
        self.neurons = neurons
        self.random_init(3, X, y)
        self.num_iters = num_iters
        self.delta = np.zeros(self.neurons * self.num_hidden_layers,)
        print(self.theta)


    #randomly initialize theta based on symmetry breaking between range:[-epsion, epsilon]
    def random_init(self, epsilon, X, y):
        theta_num = (X.shape[1] + 1) * self.neurons
        theta_num += (self.neurons + 1) * self.neurons * (self.num_hidden_layers - 1)
        theta_num += (self.neurons + 1) * y.shape[1]
        self.theta = np.random.uniform(-epsilon, epsilon, (1, theta_num))[0]
        self.delta = np.zeros(1, theta_num)

    def setData(self, training_set, validation_set, test_set):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def get_activations(self, X, theta):
        z = X * theta.T
        return np.concatenate(np.ones(z.shape[0]).T, self.sigmoid(z))

    # implement the back propagation algorithm once,
    # it returns the delta matrix for the input training example
    def back_propagation(self, X, y):
        # define variables
        #  - steps
        input_step = (X.shape[1] + 1) + self.neurons
        layer_step = (self.neurons + 1) * self.neurons
        output_step = (self.neurons + 1) * y.shape[1]
        # - activations and output
        activations = np.zeros(((self.neurons + 1) * self.num_hidden_layers))
        output = np.zeros(y.shape[1],)
        # - delta/errors
        delta_matrix = self.get_init_delta(X)

        # step1: do the forward propagation to get the delta of output,
        #        and store the activations for back propagation
        for i in range(0, self.num_hidden_layers):
            # For each if, get cur_theta by slicing self.theta,
            #   and get input from last layer by slicing activations
            if i == 0:
                cur_theta = self.theta[0: input_step].reshape((self.neurons, x.shape[1] + 1))
                activations[0, self.neurons + 1] = self.get_activations(X, cur_theta)
            elif i == self.num_hidden_layers  - 1:
                cur_theta = self.theta[-output_step : ].reshape((y.shape[1], self.neurons + 1))
                prev_activation = activations[-output_step:]
                output = self.get_activations(prev_activation, cur_theta)
            else:
                cur_theta = self.theta[input_step + (i - 1) * layer_step : input_step + i * layer_step].reshape((self.neurons, self.neurons + 1))
                prev_actiavtion = activations[layer_step * (i - 1), layer_step * i]
                activations[layer_step * i: layer_step * (i + 1)] = self.get_actiavtion(prev_activation, cur_theta)

        # step2: update delta_matrix
        # TODO: rewrite following part to implement the unrolled theta
        for i in range(2, self.num_hidden_layers):
            layer = self.num_hidden_layers - i
            coeff = self.theta[layer].T * delta_matrix[layer + 1]
            deri = np.dot(activations[layer], (1 - activations[layer]))
            delta_matrix[layer] = np.dot(coeff, deri)
        return delta_matrix

    #Here the train function should also in charge of finding an optimize solution
    # for each Theta, leave this in main function by using fmin_cg in scipy
    def train(self, X, y):
        delta_matrix= self.get_init_delta(X)
        # num_examples
        m = X.shape()[1]
        for i in range(0, m):
            delta_matrix += self.back_propagation(X.T[i].T, y[i])

