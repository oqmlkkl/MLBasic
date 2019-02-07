from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()

X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, random_state=1)

print(X_train.shape)
print(Y_train.shape)

#Use scalar to fit the X_train

scaler.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
