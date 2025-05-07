
import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset for XOR
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output dataset for XOR
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed for reproducibility
np.random.seed(1)

# Initialize weights randomly with mean 0
input_layer_neurons = X.shape[1]
hidden_layer_neurons = 2
output_neurons = 1

# Weights and biases
wh = 2 * np.random.random((input_layer_neurons, hidden_layer_neurons)) - 1
bh = np.zeros((1, hidden_layer_neurons))
wo = 2 * np.random.random((hidden_layer_neurons, output_neurons)) - 1
bo = np.zeros((1, output_neurons))

# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    # Forward Propagation
    zh = np.dot(X, wh) + bh
    ah = sigmoid(zh)

    zo = np.dot(ah, wo) + bo
    ao = sigmoid(zo)

    # Backpropagation
    error = y - ao
    d_output = error * sigmoid_derivative(ao)

    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_derivative(ah)

    # Updating Weights and Biases
    wo += ah.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss}")

# Final output
print("Final Output after Training:")
print(ao)

# Interactive prediction
def predict(x1, x2):
    x = np.array([[x1, x2]])
    zh = np.dot(x, wh) + bh
    ah = sigmoid(zh)
    zo = np.dot(ah, wo) + bo
    ao = sigmoid(zo)
    return ao

# Example prediction
print("Prediction for [0, 1]:", predict(0, 1))
