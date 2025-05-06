""" 
    Regression Model: Predicts scaler
    Predicts resting heart pressure based on age, sex, cholestrol
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages


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


    def gradient_descent(self, X, y):
        '''
            evaluates the model using gradient descent:
            Initialize w, e, a(), k = 0
            while || d/dw RSS(w) || > e:
                k += 1
                w = w - a(k) * (X^TX)^-1 X^Ty
        '''
        learning_rate = 1e-5
        iterations = 1000
        self.weights = np.random.rand(X.shape[1])
        rss_history = []
        m = X.shape[0]

        for _ in range(iterations):
            y_pred = self.predict(X)
            gradients = (2 / m) * X.T.dot(y_pred - y)
            self.weights -= (learning_rate) * gradients
            rss = self.accuracy(X,y_pred)
            rss_history.append(rss)
        
        # --- Animation ---
        fig, ax = plt.subplots()
        ax.set_xlim(0, len(rss_history))
        ax.set_ylim(0, max(rss_history))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RSS")
        ax.set_title("RSS over Iterations")
        line, = ax.plot([], [], lw=2)

        def update(frame):
            xdata = list(range(frame))
            ydata = rss_history[:frame]
            line.set_data(xdata, ydata)
            return line,

        ani = animation.FuncAnimation(fig, update, frames=len(rss_history), blit=True, interval=10)
        plt.show()


    
    def accuracy(self, X, y):
        '''
            evaluates the model using RSS
            RSS(w) = (y - Xw)^t (y - Xw)
        '''
        residual = (y - (X @ self.weights))
        rss = residual.T @ residual
#        print(f"RSS:\t{(rss[0,0])}")
#        tss = np.sum((y - np.mean(y)) ** 2)
#        r2_manual = 1 - (rss / tss)
#        print(f"R^2:\t{r2_manual[0,0]}")
        return rss  #, r2_manual[0,0]

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
        print(f"Training X : {trainX.shape}")
        print(trainX)
        print(f"Training Y : {trainY.shape}")
        print(trainY)
        
        X = np.c_[np.ones(trainX.shape[0]), trainX]
        print(f"Bias Term: {X.shape}")
        print(X)
        trainY = trainY.flatten()
        print(trainY)

        model = Regressor()
        model2 = Regressor()
        model.gradient_descent(X, trainY)
        model2.fit(X, trainY)
        print("-- gradient descent --")
        print(model.accuracy(X, trainY))
        print("- D RSS/DW-")
        print(model2.accuracy(X, trainY))

        # plot cholestrol vs resting bp
        Z = model.predict(X)
        Z2 = model2.predict(X)

        print(' - training -')
        print(trainY)
        print(' - preds -')
        print(Z)
        print('- weights -')
        print(model.weights)

        output_path = "regressor_results.pdf"
        with PdfPages(output_path) as pdf:
            plt.figure()
            plt.title("Gradient Descent")
            plt.scatter(X[:, 1], trainY, color="blue", marker="o")
            plt.plot(X[:, 1], Z, color="red", linestyle="-", linewidth=2, label="Regression Line")
            plt.xlabel("Cholestrol")
            plt.ylabel("Resting Blood Pressure")
            plt.legend()
            pdf.savefig()
            plt.clf()

            plt.figure()
            plt.title("dRSS(w)/dw")
            plt.scatter(X[:, 1], trainY, color="blue", marker="o")
            plt.plot(X[:, 1], Z2, color="red", linestyle="-", linewidth=2, label="Regression Line")
            plt.xlabel("Cholestrol")
            plt.ylabel("Resting Blood Pressure")
            plt.legend()
            pdf.savefig()
            plt.clf()



#            ---plotting for 3d features [age, cholestrol] : [resting bp s]
#             cholerstrol_axis = np.linspace(0, max(X[:,1]), len(X))
#             age_axis = np.linspace(0, max(X[:,2]), len(X))
#             bias = np.ones(len(X))
#             reg_line = model.predict(np.c_[bias, cholerstrol_axis, age_axis])


#             fig = plt.figure()
#             ax = fig.add_subplot(projection='3d')

#             ax.scatter(X[:,1], X[:,2], trainY, color="blue", marker="o")

#             ax.plot_surface(cholerstrol_axis, age_axis, reg_line)
            
#             ax.set_xlabel("Cholestrol")
#             ax.set_ylabel("Age")
#             ax.set_zlabel("Blood Pressure")
# #            ax.text(0,0,1, text, fontsize=12)
#             pdf.savefig()