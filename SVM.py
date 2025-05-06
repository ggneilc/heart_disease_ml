import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)  # convert 0 → -1

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # No hinge loss
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Misclassified or on margin
                    dw = 2 * self.lambda_param * self.w - y[idx] * x_i
                    db = -y[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)
    
    def accuracy(self, X, y):
        ''' evaluate performance : correct / total '''
        preds = self.predict(X)
        return np.mean(preds == y)

import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    y = y*2 -1
    y = y.flatten()
    def get_line(w, b, x_vals):
        return -(w[0] * x_vals + b) / w[1]

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1")
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label="Class -1")

    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = get_line(model.w, model.b, x_vals)
    plt.plot(x_vals, y_vals, 'k--')

    # Margins
    margin = 1 / np.linalg.norm(model.w)
    y_margin_up = get_line(model.w, model.b - margin, x_vals)
    y_margin_down = get_line(model.w, model.b + margin, x_vals)
    plt.plot(x_vals, y_margin_up, 'g--', linewidth=0.5)
    plt.plot(x_vals, y_margin_down, 'g--', linewidth=0.5)

    plt.legend()
    plt.title("Linear SVM Decision Boundary")
    plt.show()
