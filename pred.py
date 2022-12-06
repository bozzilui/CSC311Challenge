import re
import pandas as pd
import sys
import csv
import numpy as np
from BernouilliNBmodel import BernoulliNB
import DataTransform

file_name = "clean_quercus.csv"

if __name__ == "__main__":

    # split our training data into x and y
    x, y = DataTransform.transform_data(file_name, "train")

    x_train = x
    y_train = y

    # Create the model
    model = BernoulliNB()

    # Fit the model to the training data
    model.fit(x_train, y_train)

    if len(sys.argv) < 2:
        print("""
    Usage:
        python example_pred.py <test_data.csv>
    
    As a first example, try running `python example_pred.py example_test_set.csv`
    """)
        exit()

    # store the name of the file containing the test data
    file_name = sys.argv[-1]

    # read the file containing the test data
    # create x using test data
    x_test = DataTransform.transform_data(file_name, "test")

    pred = model.predict(x_test)

    print(pred)



