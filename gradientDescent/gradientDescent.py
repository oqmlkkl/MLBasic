import numpy as np

def computeCost(X, y, theta):
    resultMatrix = np.multiply(X, theta) - y
    return np.multiply(resultMatrix, resultMatrix).sum / (2 * m)

#theta should be a row matrix
def gradientDescent(X, y, theta, alpha, num_iters):
    #length of y
    m = y.shape[1]

    #old values of J, the result
    J_history = []

    for i in range(1: num_iters):
        thetaOld = theta
        # here using a linear model for now
        diff = np.multiply(theta, X) - y
        theta = theta - alpha * diff.sum() / m
        J_history = computeCost(X, y, theta)

    return theta, J_history
