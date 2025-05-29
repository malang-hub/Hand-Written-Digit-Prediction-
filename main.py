from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialize import initialise
from Prediction import predict
from scipy.optimize import minimize

# Loading the dataset
data = loadmat('mnist-original.mat')
X = data['data'].T / 255.0  # Normalizing
y = data['label'].flatten()

# Splitting into training and testing sets
X_train, y_train = X[:60000, :], y[:60000]
X_test, y_test = X[60000:, :], y[60000:]

# Neural network configuration
input_layer_size = 784
hidden_layer_size = 100
num_labels = 10
lambda_reg = 0.1
maxiter = 100

# Initialize weights
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])
args = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# Train neural network
result = minimize(neural_network, x0=initial_nn_params, args=args,
                  method='L-BFGS-B', jac=True, options={'maxiter': maxiter, 'disp': True})
nn_params = result.x

# Reshape Theta1 and Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, hidden_layer_size + 1))

# Evaluate model
train_pred = predict(Theta1, Theta2, X_train)
test_pred = predict(Theta1, Theta2, X_test)
print("Training Set Accuracy: {:.2f}%".format(np.mean(train_pred == y_train) * 100))
print("Test Set Accuracy: {:.2f}%".format(np.mean(test_pred == y_test) * 100))

# Precision
true_positive = np.sum(train_pred == y_train)
false_positive = len(y_train) - true_positive
print("Precision: {:.2f}".format(true_positive / (true_positive + false_positive)))

# Save weights
np.savetxt('Theta1.txt', Theta1)
np.savetxt('Theta2.txt', Theta2)