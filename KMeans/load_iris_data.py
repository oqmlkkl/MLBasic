from sklearn import datasets
"""
path = './../../../dataset/enron/enron_mail_20150507.tar.gz'
with tarfile.open(path) as archive:
    for member in archive:
        if member.isdir() and member.name.count('/') < 3:
            print(member.name)
"""

iris = datasets.load_iris()
X = iris.data[:, :2]