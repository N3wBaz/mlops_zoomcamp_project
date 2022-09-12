import os
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    return df


def save_dataframe(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)


def outlier_thresholds(
    df: pd.DataFrame, col_name: str, q1: float = 0.25, q3: float = 0.90
):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(df: pd.DataFrame, variable: str):
    low_limit, up_limit = outlier_thresholds(df, variable)
    df.loc[(df[variable] < low_limit), variable] = low_limit
    df.loc[(df[variable] > up_limit), variable] = up_limit
    return df


def replace_missing_values(data: pd.DataFrame, column: str):
    case_zero = (data['Outcome'] == 0) & (data[column].isnull())
    data.loc[case_zero, column] = data.groupby('Outcome')[column].median()[0]
    case_one = (data['Outcome'] == 1) & (data[column].isnull())
    data.loc[case_one, column] = data.groupby('Outcome')[column].median()[1]
    return data


def preprocess(df: pd.DataFrame):
    # remove zero values
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_zero] = df[cols_zero].replace(0, np.NaN)
    for col in cols_zero:
        df = replace_missing_values(df, col)

    #  remove outliers
    cols = [
        "Insulin",
        "Pregnancies",
        "SkinThickness",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    for col in cols:
        replace_with_thresholds(df, col)
    return df


def preprocess_run(raw_data_path: str, dest_path: str) -> None:
    #  load csv file

    df = read_dataframe(os.path.join(raw_data_path, "diabetes.csv"))

    #  preprocess data
    #  remove zero values, outliers and split data to test and train
    X_train, X_valid = train_test_split(
        df, test_size=0.2, stratify=df['Outcome'], random_state=42
    )
    X_train = preprocess(X_train.reset_index())
    X_valid = preprocess(X_valid.reset_index())
    # X_target = preprocess(df.reset_index())
    save_dataframe(X_train, os.path.join(dest_path, "train.csv"))
    save_dataframe(X_valid, os.path.join(dest_path, "valid.csv"))
    # save_dataframe(X_target, os.path.join(dest_path, "target.csv"))

    print("Data preprocessing is complete")


def main_preprocess():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        default="data",
        help="the location where the raw NYC taxi trip data was saved",
    )
    parser.add_argument(
        "--dest_path",
        default="data",
        help="the location where the resulting files will be saved.",
    )
    args = parser.parse_args()

    preprocess_run(args.raw_data_path, args.dest_path)


if __name__ == '__main__':
    main_preprocess()
