import numpy as np
from scipy.optimize import fmin

class LogisticRegression:

    alpha = 0.005
    #hTheta = 0

    def __init__(self, alpha):
        self.alpha = alpha

    #Why we need sigmoid funtion:
    # In classification problems(start as binary), we need to
    # decide if a data lies in class A or B (represented as 0 and 1),
    # here sigmoid funtion gives us a interval between 0
    # and 1 and distributed evenly between 0 and 1
    def sigmoid(z):
        return 1/(1 + math.exp(-z))

    def setHTheta(self, theta, X, y):
        self.hTheta = self.sigmoid(np.dot(X, theta))

    #Define the gradient of cost function of logistic regression.
    def getGrad(self, theta, X, y):
        #hTheta is the result of hypothesis function with given theta
        hTheta = np.apply_along_axis(self.sigmoid, 1, np.dot(X, theta))
        diff = hTheta - y
        gradient = np.dot(np.transpose(X), diff).sum() * self.alpha / X.shape[0]
        return gradient

    #Define the cost function
    def getCost(self, theta, X, y):
        hTheta = np.apply_along_axis(self.sigmoid, 1, np.dot(X, theta))
        loghTheta = np.apply_along_axis(math.log, hTheta)
        cost = - np.dot(np.transpose(y), loghTheta)[0][0]
        I = np.asmatrix(np.ones(y.shape(0)))
        cost -= np.dot((I - y.transpose()),
                        np.apply_along_axis(math.log, (I.transpose() - hTheta)))[0][0]
        return cost/y.shape(0)


