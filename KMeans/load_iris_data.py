from sklearn import datasets
import matplotlib.pyplot as plt
from k_means import k_means
"""
path = './../../../dataset/enron/enron_mail_20150507.tar.gz'
with tarfile.open(path) as archive:
    for member in archive:
        if member.isdir() and member.name.count('/') < 3:
            print(member.name)
"""

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
plt.scatter(X[:,0], X[:,1])
plt.show()
classifier = k_means(3, X, y, 300, True)
