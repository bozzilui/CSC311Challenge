"""
This Python file provides some useful code for reading the training file
"clean_quercus.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 

file_name = "clean_quercus.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest 
    and standardizing return list size.
    """
    s = s.replace("3-D", '')
    s = s.replace("14-dimensional", '')
    n_list = get_number_list(s)
    n_list += [-1]*(5-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_quote_at_rank(l, i):
    """Return the quote at a certain rank in list `l`.

    Quotes are indexed starting at 1 as ordered in the survey.

    If quote is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # Clean numerics

    df["q_sell"] = df["q_sell"].apply(to_numeric).fillna(0)

    # Clean for number categories

    df["q_scary"] = df["q_scary"].apply(get_number)

    # Create quote rank categories

    df["q_quote"] = df["q_quote"].apply(get_number_list_clean)

    temp_names = []
    for i in range(1,6):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        df[col_name] = df["q_quote"].apply(lambda l: find_quote_at_rank(l, i))

    del df["q_quote"]

    # Create category indicators

    new_names = []
    for col in ["q_scary"] + temp_names:
        indicators = pd.get_dummies(df[col], prefix=col)
        new_names.extend(indicators.columns)
        df = pd.concat([df, indicators], axis=1)
        del df[col]

    # Create multi-category indicators

    for cat in ["Parents", "Siblings", "Friends", "Teacher"]:
        df[f"q_remind_{cat}"] = df["q_remind"].apply(lambda s: cat_in_s(s, cat))

    del df["q_remind"]

    # Prepare data for training - use a simple train/test split for now

    df = df[new_names + ["q_sell", "label"]]

    df = df.sample(frac=1, random_state=random_state)

    x = df.drop("label", axis=1).values
    y = pd.get_dummies(df["label"].values)

    n_train = 500

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    # Train and evaluate classifiers

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    print(f"{type(clf).__name__} train acc: {train_acc}")
    print(f"{type(clf).__name__} test acc: {test_acc}")
