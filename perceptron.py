""" 
    Linear Perceptron: h(x) = sign((w^T x) + w0)
    Takes Age, Cholestrol, etc. to predict if patient has heart disease
"""
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, max_iter):
        self.weights = []
        self.max_iter = max_iter


    def fit(self, X, y):
        # initialize weights
        self.weights = np.zeros(len(X[0]))

        # PLA
        for _ in range(self.max_iter):
            errors = 0
            for i in range(X.shape[0]):
                pred = self.weights @ X[i]
                if pred != y[i]:  # misclassification
                    self.weights += y[i] * X[i]
                    errors += 1
            if errors == 0:
                break
        
        return self.weights

    def predict(self, X):
        return np.sign(X @ self.weights)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    @staticmethod
    def test_perceptron(trainX, testX, trainY, testY):
        model = Perceptron(100)
        W = model.fit(np.c_[np.ones(trainX.shape[0]), trainX], trainY)
        acc = model.accuracy(np.c_[np.ones(trainX.shape[0]), trainX], trainY)
        # Extract correct feature indices
        b, w1, w2 = W  # W = [bias, weight1, weight2]
        print(f"Weights: {b}, {w1}, {w2}")
        print(f"Accuracy: {acc}")
        x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1  # Fix feature indexing
        x_vals = np.linspace(x_min, x_max, 100)

        y_vals = -(w1 / w2) * x_vals - (b / w2)

        plt.figure()
        # Scatter plot
        plt.scatter(trainX[:, 0], trainX[:, 1], c=trainY, edgecolors='k', label="Data Points")

        # Plot the corrected decision boundary
        plt.plot(x_vals, y_vals, 'k-', linewidth=2, label="Decision Boundary")
        plt.xlabel("resting bp")
        plt.ylabel("cholestrol")
        plt.legend()
        plt.show()
