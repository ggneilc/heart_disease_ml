""" Functions for Handling Data """
import numpy as np
import pandas as pd
import random
import math

def load_features(filepath, features=["age", "sex", "resting bp s"], target=["target"], remove=False):
    """ Reads in filepath and extracts features. 
        1. Reads csv into pandas dataframe
        2. Extracts feature columms and target column into subsets
        3. Turns subsets into numpy arrays """
    data_df = pd.read_csv(filepath)
    if remove:  # remove 0 value observations for features[0]
        columns = features+target
        sub_data = data_df[columns]
        mask = sub_data[features[0]] != 0
        processed_data = sub_data[mask]
        features = np.asarray(processed_data[features])
        labels = np.asarray(processed_data[target])
    else:       # returns observations as-is
        sub_data_df = data_df[features]
        labels_df = data_df[target]
        features = np.asarray(sub_data_df)
        labels = np.asarray(labels_df)
    return features, labels

def split_data(features, labels, reduce=False):
    """ Create training (75%) and testing (25%) data set  """
    if reduce: # trim sample to only 200 points
        trainX = []
        trainY = []
        testX = []
        testY = []
        for i in range(50):
            r = random.randint(0,len(features)-1)
            trainX.append(features[r])
            trainY.append(labels[r])
        for i in range(15):
            r = random.randint(0,len(features)-1)
            testX.append(features[r])
            testY.append(labels[r])
        return np.asarray(trainX), np.asarray(testX), np.asarray(trainY), np.asarray(testY)

    cutoff = math.floor(float(len(features)) * 0.75)
    trainX = features[:cutoff]
    testX = features[cutoff:]
    trainY = labels[:cutoff]
    testY = labels[cutoff:]
    print(f"Observations, cutoff:\t{len(features)}, {cutoff}\nTrainX:\t{len(trainX)}\nTestX:\t{len(testX)}\nTrainY:\t{len(trainY)}\nTestY:\t{len(testY)}")
    return trainX, testX, trainY, testY