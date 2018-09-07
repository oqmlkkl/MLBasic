import numpy as np
import random
import traceback
import sys

# This class provides the cost function and the function to calculate the delta
class feedforwardNN:

    #num_hidden_layers is an int represents the number of hidden layers in the NN
    #neurons is a list of int which indicates the number of neurons in each hidden
    #        layer of the NN
    def __init__(self, num_hidden_layers, neurons, training_set, learning_rate):
        exe_info = sys.exc_info()
        self.num_hidden_layers = num_hidden_layers
        self.neurons = neurons
        self.training_set = training_set
        self.weights = self.init_weights(1)
        self.biases = self.init_biases(1)
        print('Bias and weight initialization complete.')
        self.num_samples = len(training_set)
        self.learning_rate = learning_rate

    #initial weights
    def init_weights(self, epsilon):
        input_size = self.training_set[0][0].shape[0]
        output_size = self.training_set[0][1].shape[0]
        size = [input_size]
        size.extend([self.neurons] * self.num_hidden_layers)
        size.append(output_size)
        return [np.random.randn(out_neurons, in_neurons) / np.sqrt(in_neurons) for in_neurons, out_neurons in zip(size[:-1], size[1:])]

    #initialize biases
    def init_biases(self, epsilon):
        input_size = self.training_set[0][0].shape[0]
        output_size = self.training_set[0][1].shape[0]
        size = [input_size]
        size.extend([self.neurons] * self.num_hidden_layers)
        size.append(output_size)
        return [np.random.randn(neuron, 1) for neuron in size[1:]]
     #==== end of init methods ==

    # using sigmoid function to pass activations.
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # gets the output of the neural network given activation is the input layer
    def get_output(self, activation):
        for bias, weight in zip(self.biases, self.weights):
            activation = self.sigmoid(np.dot(weight, activation) + bias)
        return activation

    """Update weights according to nabla given by back propagation.
       While reg is the regression parameter.
       """
    def update_weights(self, batch, reg):
        delta_bias = [np.zeros(bias.shape) for bias in self.biases]
        delta_weight = [np.zeros(weight.shape) for weight in self.weights]
        for train_x, train_y in batch:
            nabla_delta_bias, nabla_delta_weight = self.back_propagation(train_x, train_y)
            delta_bias = [ db + ndb for db, ndb in zip(delta_bias, nabla_delta_bias) ]
            delta_weight = [ dw + ndw for dw, ndw in zip(delta_weight, nabla_delta_weight) ]
        self.weights = [(1 - self.learning_rate * reg / len(batch)) * w  - self.learning_rate / len(batch) * dw for w, dw in zip(self.weights, delta_weight) ]
        self.biases  = [b - (self.learning_rate / len(batch)) * db for b, db in zip(self.biases, delta_bias)]

    # implement the back propagation algorithm once,
    # it returns the delta matrix for the input training example
    def back_propagation(self, X, y):
        # initialize variables:
        nabla_delta_bias = [np.zeros(bias.shape) for bias in self.biases]
        nabla_delta_weights = [np.zeros(weight.shape) for weight in self.weights]
        #  - steps
        # step1: do the forward propagation to get the delta of output,
        #        and store the activations and Z for back propagation
        activation = X
        activations = [X]
        z_list = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            activation = self.sigmoid(z)
            z_list.append(z)
            activations.append(activation)
        # step2: back propagate errors
        #  First get the delta for the last layer
        delta_output = (activations[-1] - y)
        delta = delta_output #* (self.sigmoid(z_list[-1]) * (1 - self.sigmoid(z_list[-1]))) <- commented to swtich to cross-entropy
                             # we can tell from here, the delta_output is not depends on the derivation of cost function anymore,
                             # which explains why the cross-entropy cost function helps prevent slow down of training.
        nabla_delta_bias[-1] = delta
        nabla_delta_weights[-1] = np.dot(delta, activations[-2].T)
        for i in xrange(2, self.num_hidden_layers + 2):
            z = z_list[-i]
            deri = self.sigmoid(z) * (1.0 - self.sigmoid(z))
            delta = np.dot(self.weights[-i + 1].T, delta) * deri
            nabla_delta_bias[-i] = delta
            nabla_delta_weights[-i] = np.dot(delta, activations[-i - 1].T)
        return(nabla_delta_bias, nabla_delta_weights)

    def evaluate(self, test_set):
        test_results = [(np.argmax(self.get_output(x)), y) for x, y in test_set]
        return sum(int(x == y) for (x, y) in test_results)

    #==== end of  back-propagation ===
    """The main function to be called to perform training"""
    def stochastic_gradient_descent(self, num_iter, batch_size, testset = None):
        for i in xrange(num_iter):
            random.shuffle(self.training_set)
            for j in xrange(0, self.num_samples, batch_size):
                self.update_weights(self.training_set[j : batch_size + j], 5.0 / self.num_samples)
            if(testset != None):
                print "> epoch {0}: {1} / {2}".format(i, self.evaluate(testset), len(testset))

