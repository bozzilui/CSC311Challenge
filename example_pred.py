"""
This Python file is example of how your `pred.py` script should
take in the input, and produce an output. Your `pred.py` script can
use different methods to process the input data, but the format
of the input it takes and the output your script produces should be
the same.

Usage:
    example_pred.py <test_data.csv>
"""

# basic python imports are permitted
import sys
import csv
import random
# numpy and pandas are also permitted
import numpy
import pandas

def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """

    # randomly choose between the three choices: image 1, 2, vs 3.
    y = random.choice([1, 2, 3])

    # return the prediction
    return y

if __name__ == "__main__":
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
    data = csv.DictReader(open(filename))

    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)

        # print the prediction to stdout
        print(pred)


