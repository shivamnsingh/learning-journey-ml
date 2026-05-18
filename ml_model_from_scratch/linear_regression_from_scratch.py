import numpy as np


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # Number of rows and columns
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):

            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):

        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


# -------------------------------
# Example Usage
# -------------------------------

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([5, 7, 9, 11, 13])

# Create model
model = LinearRegression(lr=0.01, n_iters=1000)

# Train model
model.fit(X, y)

# Predictions
predictions = model.predict(X)

print("Predictions:", predictions)
print("Weights:", model.weights)
print("Bias:", model.bias)