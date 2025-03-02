""" Loads data, builds models, runs comparison & plots """
from helper import *
from regressor import Regressor


def main():
    data_path = "./heart_statlog_cleveland_hungary_final.csv"

    features = ["cholesterol"]
    target = ["resting bp s"]

    features, labels = load_features(data_path, features=features, target=target, remove=True)
    trainX, testX, trainY, testY = split_data(features, labels)
    Regressor.test_regressor(trainX, testX, trainY, testY)
    

if __name__ == "__main__":
    main()