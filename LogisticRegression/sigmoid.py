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
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def getH(theta, X):
        return self.sigmoid(np.dot(X, theta))


    def setHTheta(self, theta, X, y):
        self.hTheta = self.sigmoid(np.dot(X, theta))

    #Define the gradient of cost function of logistic regression.
    def getGrad(self,theta, X, y):
        #hTheta is the result of hypothesis function with given theta
''' hTheta = np.apply_along_axis(self.sigmoid, 1, np.dot(X, theta))
        diff = hTheta - y
        gradient = np.dot(np.transpose(X), diff).sum() * self.alpha / X.shape[0]
        return gradient'''
        return np.dot(X.T, self.getH(theta, X)) / y.shape[0]

    #Define the cost function
    def getCost(self, theta, X, y):
        h = self.getH(theta, X)
        logH = np.log(h)
        logOne_H = np.log(1 - h)
        return (-y * logH - (1 - y) * logOne_H).mean()

    #0.5 as default, may vary along dataset
    def predict(self, X, theta, threshold=0.5):
        value = self.sigmoid(np.dot(X, theta))
        return value >= threshold


    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iters):
            gradient = self.getGrad(self.theta, X, y)
            self.theta -= self.alpha * gradient
            print('loss = ', self.getCost(self.theta, X, y))

