import numpy as np
from scipy.special import expit

def predict(Theta1, Theta2, X):
    if X.ndim == 1:
        X = X[np.newaxis, :]
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    h1 = expit(X @ Theta1.T)
    h1 = np.concatenate([np.ones((m, 1)), h1], axis=1)
    h2 = expit(h1 @ Theta2.T)
    return np.argmax(h2, axis=1)