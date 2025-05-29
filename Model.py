import numpy as np
from scipy.special import expit

def sigmoid_gradient(z):
    return expit(z) * (1 - expit(z))

def neural_network(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda_reg):
    m = X.shape[0]

    # Reshape parameters
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))

    # One-hot encode y
    Y = np.eye(num_labels)[y.astype(int)]

    # Forward Propagation
    A1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    Z2 = A1 @ Theta1.T
    A2 = np.concatenate([np.ones((m, 1)), expit(Z2)], axis=1)
    Z3 = A2 @ Theta2.T
    A3 = expit(Z3)

    # Cost Function
    cost = -Y * np.log(A3) - (1 - Y) * np.log(1 - A3)
    J = np.sum(cost) / m
    # Regularization
    J += (lambda_reg / (2 * m)) * (np.sum(Theta1[:, 1:] ** 2) +
                                   np.sum(Theta2[:, 1:] ** 2))

    # Backpropagation
    Delta1, Delta2 = np.zeros_like(Theta1), np.zeros_like(Theta2)

    d3 = A3 - Y
    d2 = d3 @ Theta2[:, 1:] * sigmoid_gradient(Z2)

    Delta1 = d2.T @ A1
    Delta2 = d3.T @ A2

    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    # Regularization for gradients
    Theta1_grad[:, 1:] += (lambda_reg / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lambda_reg / m) * Theta2[:, 1:]

    # Unroll gradients
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    return J, grad