""" Loads data, builds models, runs comparison & plots """
from helper import *
# -- combine files into 'linear models' file ---
from regressor import Regressor
from perceptron import Perceptron


def main():
    data_path = "./heart_statlog_cleveland_hungary_final.csv"

    features_regression = ["cholesterol"]
    target_regression = ["resting bp s"]
    r_features, r_labels = load_features(data_path, features=features_regression, target=target_regression, remove=True)
    r_trainX, r_testX, r_trainY, r_testY = split_data(r_features, r_labels)
    Regressor.test_regressor(r_trainX, r_testX, r_trainY, r_testY)

    features_perceptron = ["resting bp s", "cholesterol"]
    target_perceptron = ["target"]
    p_features, p_labels = load_features(data_path, features=features_perceptron, target=target_perceptron, remove=False)
    p_trainX, p_testX, p_trainY ,p_testY = split_data(p_features, p_labels)
    Perceptron.test_perceptron(p_trainX, p_testX, p_trainY, p_testY)
    

if __name__ == "__main__":
    main()