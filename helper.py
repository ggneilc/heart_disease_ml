""" Functions for Handling Data """
import numpy as np
import pandas as pd
import math

def load_features(filepath, features=["age", "sex", "resting bp s"], target=["target"], remove=False):
    """ Reads in filepath and extracts features. 
        1. Reads csv into pandas dataframe
        2. Extracts feature columms and target column into subsets
        3. Turns subsets into numpy arrays """
    data_df = pd.read_csv(filepath)
    if remove:  # remove 0 value observations -> features is 1 numeric value
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

def split_data(features, labels):
    """ Create training and testing data set. """
    cutoff = math.floor(float(len(features)) * 0.75)
    trainX = features[:cutoff]
    testX = features[cutoff:]
    trainY = labels[:cutoff]
    testY = labels[cutoff:]
    print(f"Observations, cutoff:\t{len(features)}, {cutoff}\nTrainX:\t{len(trainX)}\nTestX:\t{len(testX)}\nTrainY:\t{len(trainY)}\nTestY:\t{len(testY)}")
    return trainX, testX, trainY, testY