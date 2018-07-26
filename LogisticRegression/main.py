import datagen as g
from sigmoid import LogisticRegression

sampleArr = g.gen(1000)
x1 = sampleArr[0]
x2 = sampleArr[1]

print('x1 = ', x1)
print('x2 = ', x2)
# combine x1 x2 as a new X, and mark them as 0, 1

sig = LogisticRegression(0.005, 100)

