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
        model.accuracy(X, trainY)

        # plot cholestrol vs resting bp
        Z = model.predict(X)

        output_path = "regressor_results.pdf"
        with PdfPages(output_path) as pdf:
            plt.figure()
            plt.scatter(X[:, 1], trainY, color="blue", marker="o")
            plt.plot(X[:, 1], Z, color="red", linestyle="-", linewidth=2, label="Regression Line")
            plt.xlabel("Cholestrol")
            plt.ylabel("Resting Blood Pressure")
            pdf.savefig()
            plt.clf()

            #---plotting for 3d features [age, sex, cholestrol]
            # Create a grid of points for visualization
            # x1_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
            # x2_grid = np.linspace(X[:, 2].min(), X[:, 2].max(), 20)
            # xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
            # # Create grid with bias term for prediction
            # X_grid = np.column_stack([np.ones(xx1.size), xx1.ravel(), xx2.ravel()])

            # y_pred = model.predict(X_grid)
            # y_pred_grid = y_pred.reshape(xx1.shape)

            # # 3D Surface plot
            # fig = plt.figure(figsize=(12, 8))
            # ax = fig.add_subplot(111, projection='3d')

            # # Plot the actual data points
            # ax.scatter(X[:, 1], X[:, 2], trainY, color='red', alpha=0.5, label='Actual data')

            # # Plot the predicted surface
            # surf = ax.plot_surface(xx1, xx2, y_pred_grid, cmap='viridis', alpha=0.8)

            # # Add a color bar
            # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            # # Label axes and add title
            # ax.set_xlabel('Feature 1')
            # ax.set_ylabel('Feature 2')
            # ax.set_zlabel('Target')
            # ax.set_title('Regression Model Output')
            # ax.legend()

            # plt.tight_layout()
            # pdf.savefig()
