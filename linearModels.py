"""
    File that contains each of the linear models
    - Linear Regression
    - Linear Perceptron
    - Logistic Regression

    X is shape [num_samples, num_features]
    y is shape [num_samples, 1]
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import Perceptron as percept
from matplotlib.backends.backend_pdf import PdfPages


class LinearRegression():
    def __init__(self, lr=0.01, epochs=1000):
        ''' Linear Regression: Predict a scalar '''
        self.weights = []
        self.lr = lr
        self.epochs = epochs
        self._models = None

    def predict(self, X):
        ''' evaluation : w^T x  [75 3][3 1]=[75 1]'''
        X = _add_bias(X)
        return X @ np.transpose(self.weights)
    
    def bagged_predict(self, X):
        ''' evaluation : 1/k w^T x'''
        result = 0
        for model in self._models:
            result += model.predict(X)
        result = result/len(self._models)
        return result
    
    def fit(self, X, y):
        ''' optimization : dRSS(w) | GD'''
        # w* = (X^T X)^-1 X^T y
        X = _add_bias(X)
        x_t = X.T
        # if np.isclose(np.linalg.det(x_t @ X), 0.0):
        #     inv = np.linalg.pinv(x_t @ X)
        # else:
        #     inv = np.linalg.inv(x_t @ X)
        inv = np.linalg.inv(x_t @ X)
        self.weights = np.transpose(inv @ x_t @ y)

    def fit_grads(self, X, y, animate=False):
        ''' optimization : Gradient Descent (1/n X.T (Xw - y))'''
        x = X
        X = _add_bias(X)  # [75 3]
        self.weights = np.random.rand( 1, X.shape[1] )  # [1 3]
        acc = []


        for _ in range(self.epochs):
            y_hat = self.predict(x)  # predict() adds bias, use original x
            gradient = (X.T @ (y_hat - y)) / X.shape[0]  # [3 75] [75 1] / 75
            gradient = self.clip_gradient(gradient)
            self.weights -= self.lr * gradient.T
            acc.append(self.accuracy(x, y))
        
        # --- Animation ---
        if animate:
            fig, ax = plt.subplots()
            ax.set_xlim(0, len(acc))
            ax.set_ylim(0, max(acc))
            ax.set_xlabel("Iteration")
            ax.set_ylabel("RSS")
            ax.set_title("RSS over Iterations")
            line, = ax.plot([], [], lw=1)

            def update(frame):
                xdata = list(range(frame))
                ydata = acc[:frame]
                line.set_data(xdata, ydata)
                return line,

            ani = animation.FuncAnimation(fig, update, frames=len(acc), blit=True, interval=10)
            plt.show()

    def fit_stochastic_gd(self, X, y, animate=False):
        ''' optimization : stochastic gradient descent '''
        x = X
        X = _add_bias(X)  # [75 3]
        self.weights = np.random.rand( 1, X.shape[1] )  # [1 3]

        for _ in range(self.epochs):
            data_point_idx = np.random.randint(0, X.shape[0])
            data_point = x[data_point_idx]
            data_point = data_point.reshape(1,2)
            data_point_2 = X[data_point_idx].reshape(1,3)

            y_hat = self.predict(data_point)  # predict() adds bias, use original x
            gradient = (data_point_2.T @ (y_hat - y[data_point_idx]))  # [3 75] [75 1] / 75
            gradient = self.clip_gradient(gradient)
            self.weights -= self.lr * gradient.T
 

    def clip_gradient(self, grad, max_norm=1.0):
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / norm)
        return grad

    def accuracy(self, X, y):
        ''' evaluate performance : RSS '''
        if self._models is not None:
            residual = y - self.bagged_predict(X)
        else:
            residual = y - self.predict(X)

        rss = residual.T @ residual
        return rss

    def results(self, X, y):
        ''' plot the decision boundary '''
        X_vis = X
        y = y.flatten()
        preds = self.predict(X_vis).flatten()

        fig, ax = plt.subplots()
        ax.scatter(X_vis[:, 0], y, color='blue', label='True y')
        ax.scatter(X_vis[:, 0], preds, color='red', marker='x', label='Predicted y')

        # If 2D input, draw line
        if X_vis.shape[1] == 1:
            x_vals = np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), 100)
            X_line = _add_bias(x_vals.reshape(-1, 1))
            y_line = (X_line @ self.weights.T).flatten()
            ax.plot(x_vals, y_line, 'k--', label='Regression Line')
        else: 
            print("only 1D feature supported")

        ax.set_title("Linear Regression Fit")
        ax.set_xlabel("x₁")
        ax.set_ylabel("y")
        ax.legend()
        plt.show()

    @staticmethod
    def bagg(X, y, k=10, m=0.632):
        ''' ensemble of regression: bootstrap aggregate'''
        models = []
        for _ in range(k):
            models.append(LinearRegression())
        
        # train models on subset m * n 
        size = int(X.shape[0] * m)
        for model in models:
            reduced_x = np.zeros((size, X.shape[1]))
            reduced_y = np.zeros((size, 1))
            for i in range(size):
                point = np.random.randint(0, size)
                reduced_x[i] = X[point]
                reduced_y[i] = y[point]
            
            model.fit(reduced_x, reduced_y)
        
        bagged_model = LinearRegression()

        bagged_model._models = models
        print(f"BAGGED REGRESSION : {bagged_model.accuracy(X, y)[0,0]:.2f} : {k} models")

class Perceptron():
    def __init__(self, kernel=None, epochs=1000, animate=False):
        self.weights = []
        self.x_train = None
        self.y_train = None
        self.kernel = kernel
        self.epochs = epochs
        self.animate = animate
        self._models = None

    def predict(self, X):
        ''' evaluation : sign(w^T x)''' 
        X = _add_bias(X)
        return np.sign(X @ np.transpose(self.weights))  # [75, 3][3, 1]
    
    def bagged_predict(self, X):
        ''' evaluation : bagged (sign(w^T x))'''
        if self._models is None:
            print("Train ensemble first!")
            return 

        result = 0  # np.zeros((X.shape[0], 1))
        for model in self._models:
            result += model.predict(X)
        
        return np.sign(result/len(self._models))



    def fit(self, X, y):
        ''' optimization : PLA '''
        x = X
        X = _add_bias(X)
        y = y * 2 - 1  # PLA needs label to be 1|-1, not 1|0

        self.weights = np.random.rand( 1, X.shape[1]) 
        updates = []

        for _ in range(self.epochs):
            errors = 0
            for i in range(X.shape[0]):  # go through each sample
                pred = np.sign(self.weights @ X[i].T)  # [1 3][1 3]
                if pred != y[i]:  # misclassification
                    self.weights += y[i] * X[i]
                    updates.append(self.weights.copy())
                    errors += 1
            if errors == 0:
                print("perfected perceptron")
                break


        if self.animate:
            self._animate_training(x, y, updates)

    def non_linear_fit(self, X, y):
        ''' optimization : kernelized perceptron'''
        if self.kernel is None:
            print("Error: Please give model a kernel with .kernel = function()")
            self.kernel = self.polynomial

        X = _add_bias(X)
        y = (y*2 - 1).flatten()
        self.x_train = X
        self.y_train = y
        n_samples = X.shape[0]
        self.weights = np.zeros(n_samples)

        for _ in range(self.epochs):
#            print(f"epoch: {e}")
            for i in range(n_samples):
                result = sum((self.weights[j] * y[j] * self.kernel(X[i], X[j]))+ self.weights[0] for j in range(n_samples)) 
            if y[i] * result <= 0:
                self.weights[i] += 1
                self.weights[0] += y[i] 
            
    def polynomial(self, x1, x2, d=2):
        ''' kernel : polynomial '''
        return (np.dot(x1, x2) + 1) ** d
    
    def rbf_kernel(self, x1, x2, gamma=0.5):
        ''' kernel : radial basis '''
        diff = x1 - x2
        return np.exp(-gamma * np.dot(diff, diff))

    def predict_kernel(self, X):
        ''' evaluation : make prediction based on supports '''
        X = _add_bias(X)
        preds = []
        for x in X:
            result = sum(
                (self.weights[j] * self.y_train[j] * self.kernel(x, self.x_train[j]))
                for j in range(len(self.weights))
            )
            preds.append(np.sign(result))
        return np.array(preds)


        # a1 ... an = 0
        # if y_i (sum a_j y^j kernel(x_j, x_i) + b) < 0:
        #  a_i += 1
        #  b += y^i

    def accuracy(self, X, y):
        ''' evaluate performance : correct / total '''
        if self._models is not None:
            preds = self.bagged_predict(X)
        elif self.kernel is not None:
            preds = self.predict_kernel(X)
        else:
            preds = self.predict(X)
        return np.mean(preds == y)

    def _animate_training(self, X, y, updates):
        y = y.flatten()
        fig, ax = plt.subplots()
        
        # plot data points
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', label='Positive')
        ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', label='Negative')
        ax.set_xlim(X[:,0].min() - 1, X[:,0].max() + 1)
        ax.set_ylim(X[:,1].min() - 1, X[:,1].max() + 1)

        line, = ax.plot([], [], 'k--', linewidth=1)
        ax.legend()

        def update(i):
            w = updates[i].flatten()
            if w[2] == 0: return line,  # avoid divide-by-zero
            x_vals = np.array(ax.get_xlim())
            y_vals = -(w[0] + w[1] * x_vals) / w[2]
            line.set_data(x_vals, y_vals)
            return line,

        anim = animation.FuncAnimation(fig, update, frames=len(updates), interval=200, blit=True)
        plt.show()

    def results(self, X, y):
        ''' plot the decision boundary '''
        y = y.flatten()
        X_vis = X
        fig, ax = plt.subplots()

        ax.scatter(X_vis[y == 1][:, 0], X_vis[y == 1][:, 1], c='b', label='Class 1')
        ax.scatter(X_vis[y == 0][:, 0], X_vis[y == 0][:, 1], c='r', label='Class 0')

        # Plot decision boundary if 2D
        if X_vis.shape[1] == 2 and len(self.weights) > 0:
            w = self.weights.flatten()
            x_vals = np.array(ax.get_xlim())
            if w[2] != 0:
                y_vals = -(w[0] + w[1] * x_vals) / w[2]
                ax.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
        else:
            print("Error: only 2D feature graph supported")

        ax.set_title("Perceptron Decision Boundary")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend()
        plt.show()
    
    def results_nonlinear(self, X, y):
        y = y.flatten()
        X_vis = X
        fig, ax = plt.subplots()

        # Scatter plot of data points
        ax.scatter(X_vis[y == 1][:, 0], X_vis[y == 1][:, 1], c='b', label='Class 1')
        ax.scatter(X_vis[y == 0][:, 0], X_vis[y == 0][:, 1], c='r', label='Class 0')

        # Draw decision boundary using kernel predictions
        if self.kernel and X.shape[1] == 2:
            # Create a meshgrid over the input space
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            preds = self.predict_kernel(grid).reshape(xx.shape)

            # Plot decision boundary contour
            ax.contourf(xx, yy, preds, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'blue'])
            ax.contour(xx, yy, preds, levels=[0], linewidths=2, colors='k')

        else:
            print("Error: Decision boundary plotting requires 2D inputs and kernel")

        ax.set_title("Kernelized Perceptron Decision Boundary")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend()
        plt.show()
    
    @staticmethod
    def bagg(X, y, testX, testY, k=10, m=0.632, sci=False):
        '''
            Trains a bagged perceptron
            k : number of models
            m : size of bootstrap
        '''
        models = []
        if sci:
            for _ in range(k):
                models.append(percept())
        else:
            for _ in range(k):
                models.append(Perceptron())
        
        # train models on subset m * n 
        size = int(X.shape[0] * m)
#        print(f"Size: {X.shape[0]} * {m} = {size}")
        for model in models:
            reduced_x = np.zeros((size, X.shape[1]))
            reduced_y = np.zeros((size, 1))
            for i in range(size):
                point = np.random.randint(0, size)
                reduced_x[i] = X[point]
                reduced_y[i] = y[point]
            
            model.fit(reduced_x, reduced_y.ravel())
        
        bagged_model = Perceptron()

        bagged_model._models = models
        print(f"BAGGED PERCEPTRON : {bagged_model.accuracy(testX, testY.ravel()) * 100:.2f}% : {k} models")


    @staticmethod
    def boost(X, y, k=10):
        ''' Adaboost perceptron'''
        # each example x in X assigned weight 1/n
        # repeat k times:
        #   train classifier h_i using w(i) by minimizing classification error
        #  

class LogisticRegression():
    def __init__(self, lr=0.01, epochs=1000, animate=False):
        self.weights = []
        self.epochs = epochs
        self.lr = lr
        self.animate = animate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_p(self, X):
        ''' evaluation : sigmoid(w^T x) '''
        return self.sigmoid(X @ np.transpose(self.weights))  # [75 3][3 1] or [1 3][3 1]
    
    def predict(self, X):
        ''' 1 if sigmoid(w^T x) >= 0.5, else -1'''
        X = _add_bias(X)
        preds = []
        for i in range(X.shape[0]):
            preds.append(1 if self.predict_p(X[i]) >= 0.5 else 0)
        return preds

    def fit_stochastic(self, X, y):
        ''' optimization : Stochastic Gradient Descent'''
        x = X
        X = _add_bias(X)  # [75, 3]
        self.weights = np.random.rand( 1, X.shape[1] )  # [1, 3]

        if not self.animate:
            for _ in range(self.epochs):
                point_idx = np.random.randint(0,X.shape[0])
                data_point = X[point_idx].reshape(1,X.shape[1])

                y_hat = self.predict_p(data_point)
                gradient = np.transpose((data_point.T @ (y_hat - y[point_idx])))  # [3 75] @ [75 1] / 75
                self.weights -= self.lr * gradient

        else:
            acc = []
            for _ in range(self.epochs):
                point_idx = np.random.randint(0,X.shape[0])
                data_point = X[point_idx].reshape(1,X.shape[1])

                y_hat = self.predict_p(data_point)
                gradient = np.transpose((data_point.T @ (y_hat - y[point_idx])))  # [3 75] @ [75 1] / 75
                self.weights -= self.lr * gradient
                acc.append(self.accuracy(x, y))
            
            # --- Animation ---
            fig, ax = plt.subplots()
            ax.set_xlim(0, len(acc))
            ax.set_ylim(0, 1)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy over Iterations")
            line, = ax.plot([], [], lw=1)

            def update(frame):
                xdata = list(range(frame))
                ydata = acc[:frame]
                line.set_data(xdata, ydata)
                return line,

            ani = animation.FuncAnimation(fig, update, frames=len(acc), blit=True, interval=10)
            plt.show()


    def fit(self, X, y):
        ''' optimization : Gradient Descent '''
        x = X  # original for accuracy
        X = _add_bias(X)  # [75, 3]
        self.weights = np.random.rand( 1, X.shape[1] )  # [1, 3]

        if not self.animate:
            for _ in range(self.epochs):
                y_hat = self.predict_p(X)
                gradient = np.transpose((X.T @ (y_hat - y)) / len(y))  # [3 75] @ [75 1] / 75
                self.weights -= self.lr * gradient

        else:
            acc = []
            for _ in range(self.epochs):
                y_hat = self.predict_p(X)
                gradient = np.transpose((X.T @ (y_hat - y)) / len(y))  # [3 75] @ [75 1] / 75
                self.weights -= self.lr * gradient
                acc.append(self.accuracy(x, y))
            
            # --- Animation ---
            fig, ax = plt.subplots()
            ax.set_xlim(0, len(acc))
            ax.set_ylim(0, 1)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy over Iterations")
            line, = ax.plot([], [], lw=1)

            def update(frame):
                xdata = list(range(frame))
                ydata = acc[:frame]
                line.set_data(xdata, ydata)
                return line,

            ani = animation.FuncAnimation(fig, update, frames=len(acc), blit=True, interval=10)
            plt.show()


    def accuracy(self, X, y):
        ''' evaluate performance :  '''
        preds = self.predict(X)
        return np.mean(preds == y)

    def results(self):
        ''' plot the decision boundary '''
        pass  # not useful in this context

def _add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]
