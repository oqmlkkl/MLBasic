import numpy as np
import random
import traceback
import sys

# This class provides the cost function and the function to calculate the delta
class feedforwardNN:

    #num_hidden_layers is an int represents the number of hidden layers in the NN
    #neurons is a list of int which indicates the number of neurons in each hidden
    #        layer of the NN
    def __init__(self, num_hidden_layers, neurons, num_iters, X, y, learning_rate, num_samples):
        exe_info = sys.exc_info()
        self.num_hidden_layers = num_hidden_layers
        self.neurons = neurons
        self.random_init(1, X, y)
        self.num_iters = num_iters
        self.delta_matrix = self.get_init_delta()
        self.learning_rate = learning_rate
        self.num_samples = num_samples
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

    def get_init_delta(self):
        return np.zeros((self.num_hidden_layers, self.neurons + 1))

    def setData(self, training_set, validation_set, test_set):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def get_activations(self, X, theta):
        z = np.matmul(X,theta.T)
        return np.hstack(([1], self.sigmoid(z)))


    def update_theta(self, theta_mat, theta_change):
        for i in range(0, theta_mat.shape[0]):
            theta_mat[i] += theta_change[i] / self.num_samples
        return theta_mat

    # implement the back propagation algorithm once,
    # it returns the delta matrix for the input training example
    def back_propagation(self, X, y):
        # define variables
        #  - steps
        input_step = (X.shape[0] + 1) * self.neurons
        layer_step = (self.neurons + 1) * self.neurons
        output_step = (self.neurons + 1) * y.shape[0]
        # - input, activations and output
        nn_input = np.hstack(([1], X))
        activations = np.zeros(((self.neurons + 1) * self.num_hidden_layers))
        output = np.zeros(y.shape[0],)

        # step1: do the forward propagation to get the delta of output,
        #        and store the activations for back propagation
        for i in range(0, self.num_hidden_layers + 1):
            # For each if, get cur_theta by slicing self.theta,
            #   and get input from last layer by slicing activations
            if i == 0:
                cur_theta = self.theta[0: input_step].reshape((self.neurons, X.shape[0] + 1))
                activations[0: self.neurons + 1] = self.get_activations(nn_input, cur_theta)
            elif i == self.num_hidden_layers:
                cur_theta = self.theta[-output_step : ].reshape((y.shape[0], self.neurons + 1))
                prev_activation = activations[- self.neurons - 1:]
                output = self.get_activations(prev_activation, cur_theta)[1:]
            else:
                cur_theta = self.theta[input_step + (i - 1) * layer_step : input_step + i * layer_step].reshape((self.neurons, self.neurons + 1))
                prev_activation = activations[(self.neurons + 1) * (i - 1): (self.neurons + 1) * i]
                activations[(self.neurons + 1) * i: (self.neurons + 1) * (i + 1)] = self.get_activations(prev_activation, cur_theta)

        print('output = ')
        print(output)
        print(' y = ')
        print(y)

        # step2: update self.delta_matrix, back propagate errors
        delta_output = output - y
        delta_input = np.zeros(X.shape[0],)
        for i in range(self.num_hidden_layers, 0, -1):
            # the last theta set
            if i == self.num_hidden_layers:
                cur_theta = self.theta[-output_step : ].reshape((y.shape[0], self.neurons + 1))
                cur_activation = activations[-self.neurons - 1:]
                delta_next = delta_output
                self.delta_matrix[i - 1] = np.matmul(cur_theta.T, delta_next) * cur_activation * (1 - cur_activation)
                #update theta based on self.delta_matrix
                grad = self.delta_matrix[i - 1] * cur_activation * self.learning_rate
                self.theta[-output_step:] = self.update_theta(cur_theta, grad).reshape((1, output_step))
            else:
                start_index = -layer_step * (self.num_hidden_layers - i + 1) - output_step
                end_index = -layer_step * (self.num_hidden_layers - i) - output_step
                cur_theta = self.theta[start_index : end_index].reshape((self.neurons, self.neurons + 1))
                cur_activation = activations[(self.neurons + 1) * (i - 1): (self.neurons + 1) * i]
                delta_next = self.delta_matrix[i]
                self.delta_matrix[i - 1][1:] =  np.matmul(delta_next, cur_theta.T) * cur_activation[1:] * (1 - cur_activation)[1:]
                #update delta
                grad = self.delta_matrix[i - 1] * cur_activation * self.learning_rate
                self.theta[start_index : end_index] = self.update_theta(cur_theta, grad).reshape((1, layer_step))

    #Here the train function should also in charge of finding an optimize solution
    # for each Theta, leave this in main function by using fmin_cg in scipy
    def train(self, X, y):
        # num_examples
        self.back_propagation(X, y)

