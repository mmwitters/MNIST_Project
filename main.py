import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
#dataset from: https://www.openml.org/d/554

X = X / 255. #performing normalization

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

row_num = 45
as_arr = np.array(X.iloc[row_num].values).reshape((28, 28))
# print(y.iloc[row_num])
#
# plt.imshow(as_arr, cmap='gray')
# plt.show()

mlp = MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-4,
                    solver='sgd', verbose=True,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)

print("Training set score MLP: ", mlp.score(X_train, y_train))
print("Test set score MLP: ", mlp.score(X_test, y_test))

svm = SVC(max_iter=35)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    svm.fit(X_train, y_train)


print("Training set score SVC: ", svm.score(X_train, y_train))
print("Test set score SVC: ", svm.score(X_test, y_test))

#TODO: implement GridSearchCV for SVC for C and gamma

