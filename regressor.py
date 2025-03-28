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
        print(f"RSS:\t{(rss[0,0])}")
        tss = np.sum((y - np.mean(y)) ** 2)
        r2_manual = 1 - (rss / tss)
        print(f"R^2:\t{r2_manual[0,0]}")
        return rss[0,0], r2_manual[0,0]

# TODO --- create oop plots with nice animations for each step --- 
# --- this will make it easier to compare every step of each model ---
#   def plot_features(self):
#   def plot_results(self):
#   def save_plot(self):

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
        
        X = np.c_[np.ones(trainX.shape[0]), trainX]

        model = Regressor()
        model.fit(X, trainY)
        rss, r2 = model.accuracy(X, trainY)

        # plot cholestrol vs resting bp
        Z = model.predict(X)

        output_path = "regressor_results.pdf"
        with PdfPages(output_path) as pdf:
            plt.figure()
            plt.scatter(X[:, 1], trainY, color="blue", marker="o")
            plt.plot(X[:, 1], Z, color="red", linestyle="-", linewidth=2, label="Regression Line")
            plt.xlabel("Cholestrol")
            plt.ylabel("Resting Blood Pressure")
            text = f"Accuracy (R²): {r2:.3f}\nRSS: {rss:.3f}"
            plt.text(min(X[:,1]), max(trainY), text, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            plt.legend()
            pdf.savefig()
            plt.clf()

            #---plotting for 3d features [age, cholestrol] : [resting bp s]
            cholerstrol_axis = np.linspace(0, max(X[:,1]), len(X))
            age_axis = np.linspace(0, max(X[:,2]), len(X))
            bias = np.ones(len(X))
            reg_line = model.predict(np.c_[bias, cholerstrol_axis, age_axis])


            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.scatter(X[:,1], X[:,2], trainY, color="blue", marker="o")

            ax.plot_surface(cholerstrol_axis, age_axis, reg_line)
            
            ax.set_xlabel("Cholestrol")
            ax.set_ylabel("Age")
            ax.set_zlabel("Blood Pressure")
            ax.text(0,0,1, text, fontsize=12)
            pdf.savefig()