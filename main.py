import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
#dataset from: https://www.openml.org/d/554

X = X / 255.

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

row_num = 45
as_arr = np.array(X.iloc[row_num].values).reshape((28, 28))
# print(y.iloc[row_num])
#
# plt.imshow(as_arr, cmap='gray')
# plt.show()

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=15, alpha=1e-4,
                    solver='sgd', verbose=True,
                    learning_rate_init=.1)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: ", mlp.score(X_train, y_train))
print("Test set score: ", mlp.score(X_test, y_test))