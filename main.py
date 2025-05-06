""" Loads data, builds models, runs comparison & plots """
from helper import *
from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# -- combine files into 'linear models' file ---
import linearModels
import SVM



def main():
    data_path = "./heart_statlog_cleveland_hungary_final.csv"

# ----- Feature Extraction -----
#    features_regression = ["cholesterol", "age", "sex", "chest pain type"]
    features_regression = ["cholesterol", "age"]
    target_regression = ["resting bp s"]
    r_features, r_labels = load_features(data_path, features=features_regression, target=target_regression, remove=True)
    r_trainX, r_testX, r_trainY, r_testY = split_data(r_features, r_labels, reduce=True)

#    features_classification = ["cholesterol", "age", "sex", "chest pain type", "resting bp s"]
    features_classification = ["resting bp s", "max heart rate"]
    target_classification = ["target"]
    p_features, p_labels = load_features(data_path, features=features_classification, target=target_classification)
    p_trainX, p_testX, p_trainY, p_testY = split_data(p_features, p_labels, reduce=False)


# ----- Model Creation & Fitting -----
    # --- Linear Regression ---
    regressor = linearModels.LinearRegression()
    regressor.fit(r_trainX, r_trainY)
    print(f"REGRESSION (dRSS(w)) : {regressor.accuracy(r_testX, r_testY)[0,0]:.2f} : {regressor.weights[0]}" )
    regressor.fit_grads(r_trainX, r_trainY)
    print(f"REGRESSION (GD) : {regressor.accuracy(r_testX, r_testY)[0,0]:.2f} : {regressor.weights[0]}" )

    # --- Linear Perceptron ---
    perceptron = linearModels.Perceptron(epochs=50, animate=False)
    perceptron.fit(p_trainX, p_trainY)
    print(f"PERCEPTRON : {perceptron.accuracy(p_testX, p_testY) * 100:.2f}% : {perceptron.weights[0]}")

  

    # --- Logistic Regression ---
    logregressor = linearModels.LogisticRegression(lr=0.0001, animate=False)
    logregressor.fit(p_trainX, p_trainY)
    print(f"LOGISTIC : {logregressor.accuracy(p_testX, p_testY) * 100:.2f}% : {logregressor.weights[0]}")

    # --- Support Vector (Hard Margin) ---
#    machine = SVM.SVM()
#    machine.fit(p_trainX, p_trainY)
#    print(f"SVM (HARD-MARGIN): {machine.accuracy(p_trainX, p_trainY) * 100:.2f}% : {machine.w}")
    

    # Those were the primitive models, now we want to improve them:
    # stochastic gradient descent, non-linear kernels
    # --- stochastic regression
    regressor.fit_stochastic_gd(r_trainX, r_trainY)
    print(f"REGRESSION (SGD) : {regressor.accuracy(r_testX, r_testY)[0,0]:.2f} : {regressor.weights[0]}" )
    logregressor.fit_stochastic(p_trainX, p_trainY)
    print(f"LOGISTIC (SGD): {logregressor.accuracy(p_testX, p_testY) * 100:.2f}% : {logregressor.weights[0]}")


    # --- Polynomial Perceptron ---
    perceptron.kernel = perceptron.polynomial
    perceptron.non_linear_fit(p_trainX, p_trainY)
    print(f"PERCEPTRON (NON_LINEAR d=2) : {perceptron.accuracy(p_trainX, p_trainY) * 100:.2f}% ")
# #    perceptron.results_nonlinear(p_trainX, p_trainY)

#     # --- RBF Perceptron ---
    perceptron.kernel = perceptron.rbf_kernel
    perceptron.non_linear_fit(p_trainX, p_trainY)
    print(f"PERCEPTRON (RADIAL_BASIS) : {perceptron.accuracy(p_trainX, p_trainY) * 100:.2f}% ")
#    perceptron.results_nonlinear(p_trainX, p_trainY)


    # Then do validation testing  -> train a bunch of models and pick the best

    # Ensembles
    # --- Bagged Models ---
    linearModels.LinearRegression.bagg(r_trainX, r_trainY)
    linearModels.Perceptron.bagg(p_trainX, p_trainY, p_testX, p_testY)
    linearModels.Perceptron.bagg(p_trainX, p_trainY, p_testX, p_testY, sci=True)

    # if we can finish those, then compare with advanced models
    # logistic regression vs bayes classifier
    # perceptron/svm vs decision tree
    # MLP ? ? ? 

# ----- ----- Comparison vs scikit-learn's implementation ----- ----- #

    linreg = LinearRegression()
    linreg.fit(r_trainX, r_trainY)
    preds = linreg.predict(r_testX)
    residuals = r_testY - preds
    rss = np.sum(residuals ** 2)
    print(f"Linear Regression RSS: {rss:.2f}")

    perceptron = Perceptron()
    perceptron.fit(p_trainX, p_trainY.ravel())
    preds = perceptron.predict(p_testX)
    acc = accuracy_score(p_testY.ravel(), preds)
    print(f"Perceptron Accuracy: {acc * 100:.2f}%")

    logreg = LogisticRegression()
    logreg.fit(p_trainX, p_trainY.ravel())
    preds = logreg.predict(p_testX)
    acc = accuracy_score(p_testY.ravel(), preds)
    print(f"Logistic Regression Accuracy: {acc * 100:.2f}%")

    s_machine = SVC()
    s_machine.fit(p_trainX, p_trainY.ravel())
    preds = s_machine.predict(p_testX)
    acc = accuracy_score(p_testY.ravel(), preds)
    print(f"SVM accuracy: {acc * 100:.2f}%")

    # prettify output with ncurses & matplotlib plots


    

if __name__ == "__main__":
    main()