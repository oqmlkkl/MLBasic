import numGen as g
import matplotlib.pyplot as plt
import numpy as np
import gradientDescent as gd

n = 100

(x, y) = g.gen((n,1), [3])
xArr = np.squeeze(np.asarray(x))
yArr = np.squeeze(np.asarray(y))
#print(xArr, yArr)
plt.plot(xArr, yArr, 'ro')
plt.show()

theta = np.ones(1)
alpha = 0.005
theta = gd.gradientDescent(x, y, theta, alpha, 1)
print(theta)
