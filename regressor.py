""" 
    Regression Model: Predicts scaler
    Predicts resting heart pressure based on age, sex, cholestrol
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math


class Regressor():
    def __init__(self):
        self.weights = []

    def fit(self, X, y):
        ''' 
            finds the best fit line for the given X using dRSS(w)/dw
            w = (X^T X)^-1 X^T y
        '''
        x_t = X.T
        inv = np.linalg.inv(x_t @ X)
        self.weights = inv @ x_t @ y

    
    def accuracy(self, X, y):
        '''
            evaluates the model using RSS
            RSS(w) = (y - Xw)^t (y - Xw)
        '''
        residual = (y - (X @ self.weights))
        rss = residual.T @ residual
        print(f"RSS:\t\t{(rss[0,0])}\nsqrt(RSS):\t{math.sqrt(rss)}")



    def predict(self, X):
        ''' 
            iterate through x and multiply by self.weights
            store y for each x and return list of y's
            y_i = w^T x_i
        '''
        return X @ self.weights

    
    @staticmethod
    def test_regressor(trainX, testX, trainY, testY):
        """ Basically Main method for this class:
            Create a model, fit model, predict, graph """
        
        x = np.c_[np.ones(trainX.shape[0]), trainX]

        model = Regressor()
        model.fit(x, trainY)
        model.accuracy(x, trainY)

        # plot cholestrol vs resting bp
        Z = model.predict(x)

        output_path = "regressor_results.pdf"
        with PdfPages(output_path) as pdf:
            plt.figure()
            plt.scatter(x[:, 1], trainY, color="blue", marker="o")
            plt.plot(x[:, 1], Z, color="red", linestyle="-", linewidth=2, label="Regression Line")
            plt.xlabel("Cholestrol")
            plt.ylabel("Resting Blood Pressure")
            pdf.savefig()
            plt.clf()
