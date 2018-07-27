import datagen as g
import numpy as np
import matplotlib.pyplot as plt
from sigmoid import LogisticRegression

sampleArr = g.gen(300)
x1 = sampleArr[0]
x2 = sampleArr[1]

print('x1 = ', x1)
print('x2 = ', x2)
# combine x1 x2 as a new X, and mark them as 0, 1

X = np.append(x1[0:280], x2[0:280], axis=0)
print('X =', X)
y = np.asmatrix(np.append(np.zeros(280), np.ones(280))).T
print('y = ', y)

sig = LogisticRegression(0.005, 200)
sig.fit(X, y)

testX = np.append(x1[281: 300], x2[281:300], axis=0)
predict = sig.predict(testX)
plt.figure(figsize=(12,8))
plt.scatter(testX[:,0], testX[:,1], c=predict, alpha=.4)
plt.legend()
plt.show()



