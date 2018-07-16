import random
import numpy as np

#generate random x, y pairs according to the given shape
# shape is a tuple, and thata is an array which length == shape[1]
def gen(shape, theta):
    randX = np.random.rand(shape[0], shape[1]) * 100
    randThetaZero = np.random.uniform(low=-10, high=10, size=(shape[0], 1))
    y = sum(np.transpose(randX * theta)) + np.transpose(randThetaZero)
    return (np.asmatrix(np.round(randX)),
            np.asmatrix(np.round(np.transpose(y))))

