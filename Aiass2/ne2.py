import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your dataset
data = np.genfromtxt(r'C:\\Users\\ASUS\\Desktop\\AiAss\\Aiass2\\diabetes.csv', delimiter=',')

# Split the data into features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1]

# Split the data into train (70%), test (15%), and validation (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=45)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=45)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, max_iter, lr):
    theta = np.random.rand(X.shape[1])
    history = []

    for itr in range(1, max_iter + 1):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / len(X)
        theta -= lr * gradient
        epsilon = 1e-15
        J = (-1 / len(X)) * (np.dot(y, np.log(h + epsilon)) + np.dot(1 - y, np.log(1 - h + epsilon)))

        history.append(J)

    return theta, history

def predict(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    predicted_labels = (h >= 0.5).astype(int)
    return predicted_labels

def validate(X, y, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    predicted_labels = (h >= 0.5).astype(int)
    correct = np.sum(predicted_labels == y)
    val_accuracy = correct * 100 / len(y)
    return val_accuracy

max_iter = 200
learning_rate = 0.0001

trained_theta, history = train(X_train, y_train, max_iter, learning_rate)
val_accuracy = validate(X_val, y_val, trained_theta)
print("Validation Accuracy:", val_accuracy)

plt.plot(range(1, max_iter + 1), history, color='green')
plt.xlabel("Epoch (Number Of Iteration)")
plt.ylabel("Train Loss")
plt.title("Train Loss vs Epoch")
plt.show()
