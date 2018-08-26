import numpy as np

class feedforwardNN:
    def __init__(self, num_hidden_layers, neurons):
        try:
            self.num_hidden_layers = num_hidden_layers
            self.neurons = neurons
            if(num_hidden_layers != len(neurons)):
                raise ValueError('Cannot continue: neurons input has different length as num_hidden_layers')
        except Exception as error:
            print(error)

    def setData(self, training_set, validation_set, test_set):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def activation(self, X, theta):
        z = X * theta.T
        return np.concatenate(np.ones(z.shape[0]).T, self.sigmoid(z))

    def

    def cost():
