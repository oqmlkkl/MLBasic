import numpy as np

def computeCost(X, y, theta, m):
    resultMatrix = np.multiply(X, theta) - y
    return np.multiply(resultMatrix, resultMatrix).sum() / (2 * m)

#theta should be a row matrix
def gradientDescent(X, y, theta, alpha, num_iters):
    #length of y
    m = y.shape[0]

    const = np.ones((m, 1))
    X = np.append(const, X, axis = 1)
    print('X = ', X)
    #only intialize the const
    theta = np.insert(theta, 1,1,axis = 0)
    print('theta = ', theta)

    #use J_history to mark the cost
    J_history = 0

    for i in range(0, num_iters):
        thetaOld = theta
        # here using a linear model for now
        hypothesis = np.multiply(theta, X)
        diff = hypothesis - y
        print('theta = ', theta)

        #update theta0
        gradient = alpha * diff.sum() / m
        theta = theta - np.insert(np.zeros(y.shape[1]), 0, gradient, axis = 0)
        #update the rest theta
        #sleepy leave it tomorrow
        J_history = computeCost(X, y, theta, m)

    return theta
