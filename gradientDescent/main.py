import numGen as g
import matplotlib.pyplot as plt
import numpy as np
import gradientDescent as gd

n = 100
thetaArr = [3, 5]
(x, y) = g.gen((n,2), thetaArr)
xArr = np.squeeze(np.asarray(x))
yArr = np.squeeze(np.asarray(y))
#print(xArr, yArr)
plt.plot(xArr, yArr, 'ro')
plt.show()

alpha = 0.005
theta = gd.gradientDescent(x, y, alpha, 2000)
print('Designed theta: ', thetaArr)
print('Trained theta: ', theta[:,[1, theta.shape[1] - 1]])
