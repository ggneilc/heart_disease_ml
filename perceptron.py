""" 
    Linear Perceptron: h(x) = sign((w^T x) + w0)
    Takes Age, Cholestrol, etc. to predict if patient has heart disease
"""
import numpy as np

class Perceptron():
    def __init__(self):
        self.weights = []


    def fit(self, X, y):
        pass


    def predict(self, X):
        pass

    @staticmethod
    def test_perceptron(trainX, testX, trainY, testY):
        model = Perceptron()
        x = np.c_[np.ones(trainX.shape[0]), trainX]
        model.fit(x, trainY)
