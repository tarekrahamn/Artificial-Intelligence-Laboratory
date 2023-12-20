import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt(r'C:\\Users\\ASUS\\Desktop\\AiAss\\Aiass2\\diabetes.csv', delimiter=',')
max_iter = 500
learning_rates = [0.1, 0.01, 0.001, 0.0001]

# Make sure you have Train, Trtarget, Val, and Vtarget defined.

def train(X, y, max_iter, lr):
    theta = np.random.rand(len(X[0]))  # Removed unnecessary parentheses
    history = []
    
    for itr in range(1, max_iter+1):
        TJ = 0
        for i in range(len(X)):
            feature = X[i].astype(float)
            z = np.dot(feature, np.array(theta, dtype=float))
            h = (1 / (1 + np.exp(-z)))
            J = -feature * np.log(h) - (1 - y[i]) * np.log(1 - h)  # Removed unnecessary type conversion
            TJ += J
            feature_array = X[i].astype(float)
            diff_array = (h - y[i]).astype(float)
            dv = np.dot(feature_array, diff_array)

            theta -= lr * dv

        TJ /= len(X)
        history.append(TJ)
    
    return theta, history

def validate(X, y, theta):
    correct = 0

    for i in range(len(X)):
        feature = X[i].astype(float)
        z = np.dot(feature, np.array(theta, dtype=float))
        h = (1 / (1 + np.exp(-z)))
        predicted_label = 1 if h >= 0.5 else 0
        if predicted_label == y[i]:
            correct += 1

    val_acc = correct * 100 / len(X)
    
    return val_acc

trained_theta = 0
Btheta = 0
results = []

# You need to define Train, Trtarget, Val, and Vtarget before using them in the following loop.

for lr in learning_rates:
    trained_theta, history = train(Train, Trtarget, max_iter, lr)
    val_acc = validate(Val, Vtarget, trained_theta)
    results.append((lr, val_acc, trained_theta))

print(results)

# %matplotlib inline (this line is not needed in regular Python scripts)

plt.plot(range(1, max_iter + 1), history, color='green')
plt.xlabel("Epoch (Iteration)")
plt.ylabel("Train Loss")
plt.title("Train Loss vs Epoch")
plt.show()
