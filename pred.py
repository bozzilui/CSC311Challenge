import re
import pandas as pd
import sys
import csv
import numpy as np
from BernouilliNBmodel import BernoulliNB
import DataTransform


if __name__ == "__main__":
    x, y = DataTransform.x, DataTransform.y

    n_train = 500

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    model = BernoulliNB()

    # Fit the model to the training data
    model.fit(x_train, y_train)

    # check if the argument <test_data.csv> is provided
    if len(sys.argv) < 2:
        print("""
    Usage:
        python example_pred.py <test_data.csv>

    As a first example, try running `python example_pred.py example_test_set.csv`
    """)
        exit()

    # store the name of the file containing the test data
    filename = sys.argv[-1]

    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = pd.read_csv(filename)

    for test_example in data:
        # obtain a prediction for this test example
        pred = model.predict(test_example)

        # print the prediction to stdout
        print(pred)

    pred = model.predict(x_test)

    # Print the predicted class labels
    print(np.mean((y_test-pred)^2))


