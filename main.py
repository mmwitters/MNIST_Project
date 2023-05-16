import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
#dataset from: https://www.openml.org/search?type=data&sort=runs&id=554&status=active

X = X / 255.

# rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

as_arr = np.array(X.iloc[0])

img_plt = plt.imshow(as_arr.reshape(28,28), cmap='gray')