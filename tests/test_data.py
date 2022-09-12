from pathlib import Path

import numpy as np
import pandas as pd


def read_data(file):
    test_directory = Path(__file__).cwd()
    return pd.read_csv(test_directory / 'data' / file)


def test_column_names():

    ans = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age',
        'Outcome',
    ]
    # data = read_dataframe("tests/test.csv")
    # data = read_dataframe("./data/diabetes.csv")
    data = read_data("diabetes.csv")
    names = [1 if i in ans else 0 for i in data.columns.values]
    if 0 in names:
        names = 0
    else:
        names = 1
    assert names == 1


def test_values():
    # pylint: disable=unsubscriptable-object
    data = read_data("diabetes.csv")
    number_type = []
    rignt_type = [
        np.int64,
        np.int64,
        np.int64,
        np.int64,
        np.int64,
        np.float64,
        np.float64,
        np.int64,
        np.int64,
    ]
    for i in data.columns.values:
        number_type.append(type(data[i][0]))
    n = 1
    for i, j in zip(number_type, rignt_type):
        if i != j:
            n = 0
    assert n == 1
