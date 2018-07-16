import numpy as np

def computeCost(X, y, theta, m):
    resultMatrix = np.multiply(X, theta) - y
    return np.multiply(resultMatrix, resultMatrix).sum() / (2 * m)

#theta should be a row matrix
def gradientDescent(X, y, alpha, num_iters):
    #length of y
    m = y.shape[0]

    const = np.ones((m, 1))
    X = np.append(const, X, axis = 1)
    #initialize theta
    theta = np.asmatrix(np.ones(X.shape[1]))

    for i in range(0, num_iters):
        thetaOld = theta
        #update hypothesis
        hypothesis = np.dot(X, np.transpose(theta))
        diff = hypothesis - y
        #get gradient
        gradient = np.dot(np.transpose(diff), X) / (2 * m)
        theta = theta - alpha * gradient
        print('theta = ', theta)

    return theta
