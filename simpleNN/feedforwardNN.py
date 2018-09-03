import numpy as np
import random
import traceback
import sys

# This class provides the cost function and the function to calculate the delta
class feedforwardNN:

    #num_hidden_layers is an int represents the number of hidden layers in the NN
    #neurons is a list of int which indicates the number of neurons in each hidden
    #        layer of the NN
    def __init__(self, num_hidden_layers, neurons, num_iters, X, y, learning_rate):
        exe_info = sys.exc_info()
        self.num_hidden_layers = num_hidden_layers
        self.neurons = neurons
        self.random_init(3, X, y)
        self.num_iters = num_iters
        self.delta = np.zeros(self.neurons * self.num_hidden_layers,)
        self.learning_rate = learning_rate
        print(self.theta)

    def update_theta(self, detla_matrix, x_dim, y_dim):
        input_theta = self.theta[0, (x_dim + 1) * self.neurons].reshape(self.neurons, x_dim + 1)
        hidden_layer_theta = self.theta[(x_dim + 1) * self.neurons, (self.neurons + 1) * self.neurons * (self.num_hidden_layers - 1)].reshape(self.num_hidden_layers - 1, self.neurons, self.neurons + 1)
        output_theta = self.theta[-(self.neurons + 1) * y_dim].reshape(y_dim, self.neurons + 1)

    #randomly initialize theta based on symmetry breaking between range:[-epsion, epsilon]
    def random_init(self, epsilon, X, y):
        theta_num = (X.shape[1] + 1) * self.neurons
        theta_num += (self.neurons + 1) * self.neurons * (self.num_hidden_layers - 1)
        theta_num += (self.neurons + 1) * y.shape[1]
        self.theta = np.random.uniform(-epsilon, epsilon, (1, theta_num))[0]

    def get_init_delta(self, X):
        return np.zeros((self.num_hidden_layers, self.neurons + 1))

    def setData(self, training_set, validation_set, test_set):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def get_activations(self, X, theta):
        z = X * theta.T
        return np.concatenate(np.ones(z.shape[0]).T, self.sigmoid(z))

    def update_theta(theta_mat, theta_change):
        for i in theta_mat.shape[0]:
            theta_mat[i] += theta_change[i]
        return theta_mat

    # implement the back propagation algorithm once,
    # it returns the delta matrix for the input training example
    def back_propagation(self, X, y):
        # define variables
        #  - steps
        input_step = (X.shape[1] + 1) + self.neurons
        layer_step = (self.neurons + 1) * self.neurons
        output_step = (self.neurons + 1) * y.shape[1]
        # - input, activations and output
        nn_input = np.hstack(([1], X))
        activations = np.zeros(((self.neurons + 1) * self.num_hidden_layers))
        output = np.zeros(y.shape[1],)
        # - delta/errors
        delta_matrix = self.get_init_delta(X)

        # step1: do the forward propagation to get the delta of output,
        #        and store the activations for back propagation
        for i in range(0, self.num_hidden_layers + 1):
            # For each if, get cur_theta by slicing self.theta,
            #   and get input from last layer by slicing activations
            if i == 0:
                cur_theta = self.theta[0: input_step].reshape((self.neurons, X.shape[1] + 1))
                activations[0, self.neurons + 1] = self.get_activations(nn_input, cur_theta)
            elif i == self.num_hidden_layers:
                cur_theta = self.theta[-output_step : ].reshape((y.shape[1], self.neurons + 1))
                prev_activation = activations[-output_step:]
                output = self.get_activations(prev_activation, cur_theta)
            else:
                cur_theta = self.theta[input_step + (i - 1) * layer_step : input_step + i * layer_step].reshape((self.neurons, self.neurons + 1))
                prev_actiavtion = activations[layer_step * (i - 1), layer_step * i]
                activations[layer_step * i: layer_step * (i + 1)] = self.get_actiavtion(prev_activation, cur_theta)

        # step2: update delta_matrix, back propagate errors
        delta_output = output - y
        delta_input = np.zeros(X.shape[1],)
        for i in range(self.num_hidden_layers, -1, -1):
            # the last theta set
            if i == self.num_hidden_layers:
                cur_theta = self.theta[-output_step : ].reshape((y.shape[1], self.neurons + 1))
                cur_activation = activations[-output_step:]
                delta_next = delta_output
                delta_matrix[i] = np.matmul(cur_theta.T, delta_next) * cur_activation * (1 - cur_activation)
                #update theta based on delta_matrix
                grad = delta_matrix[i] * cur_activation * self.learning_rate
                self.theta[-output_step:] = self.update_theta(cur_theta, grad).reshape((1, output_step))
            # update delta for input layer
            elif i == 0:
                cur_theta = self.theta[0 : input_step].reshape((self.neurons,  X.shape[1] + 1))
                cur_activation = nn_input
                delta_next = delta_matrix[0]
                delta_input = np.matmul(cur_theta.T, delta_next) * cur_activation * (1 - cur_activation)
            else:
                cur_theta = self.theta[-output_step * (self.num_hidden_layers - i) : -output_step * (self.num_hidden_layers - i + 1)].reshape((self.neurons, self.neurons + 1))
                cur_activation = activations[output_step * (i - 1) : output_step * i]
                delta_next = delta_matrix[i + 1]
                delta_matrix[i] =  np.matmul(cur_theta.T, delta_next) * cur_activation * (1 - cur_activation)
                #update delta
                grad = delta_matrix[i] * cur_activation * self.learning_rate
                self.theta[-output_step * (self.num_hidden_layers - i) : -output_step * (self.num_hidden_layers - i + 1)] = self.update_theta(cur_theta, grad)

    #Here the train function should also in charge of finding an optimize solution
    # for each Theta, leave this in main function by using fmin_cg in scipy
    def train(self, X, y):
        # num_examples
        m = X.shape[1]
        delta_matrix = self.get_init_delta()
        for i in range(0, m):
            self.back_propagation(X[i], y[i])



