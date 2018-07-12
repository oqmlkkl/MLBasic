import random
import numpy as np

#generate random x, y pairs according to the given shape
# shape is a tuple, and thata is an array which length == shape[1]
def gen(shape, theta):
    randX = np.random.rand(shape[0], shape[1])
    y = sum(np.transpose(randX * theta))
    return (randX, y)

