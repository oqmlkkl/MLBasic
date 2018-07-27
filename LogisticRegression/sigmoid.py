import numpy as np
from scipy.optimize import fmin

class LogisticRegression(object):

    def __init__(self, alpha=0.05, num_iters=100):
        self.alpha = alpha
        self.num_iters = num_iters

    #Why we need sigmoid funtion:
    # In classification problems(start as binary), we need to
    # decide if a data lies in class A or B (represented as 0 and 1),
    # here sigmoid funtion gives us a interval between 0
    # and 1 and distributed evenly between 0 and 1
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def getH(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    #Define the gradient of cost function of logistic regression.
    def getGrad(self, X, y):
        return np.dot(X.T, self.getH(X)) / y.shape[0]

    #Define the cost function
    def getCost(self, X, y):
        h = self.getH(X)
        logH = np.log(h)
        logOne_H = np.log(1 - h)
        return (-y * logH - (1 - y) * logOne_H).mean()

    #0.5 as default, may vary along dataset
    def predict(self, X, theta, threshold=0.5):
        value = self.sigmoid(np.dot(X, theta))
        return value >= threshold


    def fit(self, X, y):
        self.theta = np.ones(X.shape[1])
        for i in range(self.num_iters):
            gradient = self.getGrad(X, y)
            self.theta -= self.alpha * gradient
            print('loss = ', self.getCost(X, y))
            print('thet = ', self.theta)

    def predict(self, X):
        return self.getH(X).round()

