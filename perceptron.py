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
                pred = np.sign(self.weights @ X[i])
                if pred != y[i]:  # misclassification
                    self.weights += y[i] * X[i]
                    errors += 1
            if errors == 0:
                print("perfected perceptron")
                break
        
        
        return self.weights

    def predict(self, X):
        return np.sign(X @ self.weights)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    
    def show_result(self, X, y, W):
        """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or 0.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
        """
        y = y.ravel()
        plt.figure(figsize=(8, 6))

        # Plot data points
        for label, marker, color in zip([-1, 1], ['o', 's'], ['blue', 'red']):
            plt.scatter(X[y == label][:, 0], X[y == label][:, 1], marker=marker, color=color, label=f'Class {label}')
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        x_vals = np.linspace(x_min, x_max, 100)

        if W[2] != 0:
            y_vals = -(W[1] / W[2]) * x_vals - (W[0] / W[2])
            plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
        else:
            x_val = -W[0] / W[1]
            plt.axvline(x=x_val, color='k', linestyle='--', label='Decision Boundary')


        plt.xlabel("Feature 1 (Cholesterol)")
        plt.ylabel("Feature 2 (Resting bp s)")
        plt.legend()
        plt.title("Perceptron Decision Boundary")
        plt.savefig('result.png')
        plt.show()




    @staticmethod
    def test_perceptron(trainX, testX, trainY, testY):
        trainY = trainY * 2 - 1  # convert from {0, 1} to {-1, 1}

        model = Perceptron(100)
        W = model.fit(np.c_[np.ones(trainX.shape[0]), trainX], trainY)
        acc = model.accuracy(np.c_[np.ones(trainX.shape[0]), trainX], trainY)

        b, w1, w2 = W  # W = [bias, weight1, weight2]
        print(f"Weights: {b}, {w1}, {w2}")
        print(f"Accuracy: {acc}")

        model.show_result(trainX, trainY, W)
        
        preds = model.predict(np.c_[np.ones(trainX.shape[0]), trainX])
        model.show_result(trainX, preds, W)

