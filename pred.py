import re
import pandas as pd
import sys
import csv
import numpy as np
from BernouilliNBmodel import BernoulliNB
import DataTransform

file_name = "clean_quercus.csv"

if __name__ == "__main__":

    x, y = DataTransform.transform_data(file_name, "train")

    n_train = 500

    x_train = x[:n_train]
    y_train = y[:n_train]

    #x_test = x[n_train:]
    #y_test = y[n_train:]

    model = BernoulliNB()

    # Fit the model to the training data
    model.fit(x_train, y_train)
    """
    # check if the argument <test_data.csv> is provided
    if len(sys.argv) < 2:
        print(
    Usage:
        python example_pred.py <test_data.csv>

    As a first example, try running `python example_pred.py example_test_set.csv`
    )
        exit()

    # store the name of the file containing the test data
    filename = sys.argv[-1]
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    x_test = DataTransform.transform_data(file_name, "test")

    pred = model.predict(x_test)

    print(pred)



