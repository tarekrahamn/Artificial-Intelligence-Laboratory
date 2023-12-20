import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt(r'C:\\Users\\ASUS\\Desktop\\AiAss\\Aiass2\\diabetes.csv', delimiter=',')
# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.001, epochs=100):
    m, n = X.shape
    weights = np.random.random((1, n))
    bias = np.random.random()
    
    for epoch in range(epochs):
        z = np.dot(X, weights.T) + bias
        h = sigmoid(z)
        J = -y * np.log(h) - (1 - y) * np.log(1 - h)
        TJ = np.sum(J) / m

        dv = np.dot(X.T, (h - y)) / m
        weights -= lr * dv
        bias -= lr * np.sum(h - y) / m

        print(f'Epoch {epoch + 1}: Loss {TJ}')

    return weights, bias

# Example usage:
# Assuming you have X and y defined.
# X = ... # Your feature matrix
# y = ... # Your target values (0 or 1)

# Train the model
trained_weights, trained_bias = train(X, y, lr=0.001, epochs=100)


# Define the predict function
def predict(X, weights, bias):
    z = np.dot(X, weights.T) + bias
    h = sigmoid(z)
    predictions = (h >= 0.5).astype(int)
    return predictions

train_size = int(0.7 * data.shape[0])
# Define the size of the testing set
test_size = int(0.15 * data.shape[0])  # 15% of the data for testing
# Split data into train(70%), test(15%), and validation(15%)
# Make sure you have X_train, y_train, X_test, y_test, X_val, y_val defined
X_train = data[:train_size, :-1]
y_train = data[:train_size, -1]
X_test = data[train_size:train_size + test_size, :-1]
y_test = data[train_size:train_size + test_size, -1]
X_val = data[train_size + test_size:, :-1]
y_val = data[train_size + test_size:, -1]
# Train the model
trained_weights, trained_bias, loss_values = train(X_train, y_train, lr=0.0001, epochs=200)

# Evaluate the model on the validation set
validation_predictions = predict(X_val, trained_weights, trained_bias)
validation_accuracy = np.mean(validation_predictions == y_val)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')

# Plot the loss curve
plt.plot(range(1, len(loss_values) + 1), loss_values)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
