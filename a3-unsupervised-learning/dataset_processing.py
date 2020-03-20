import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve

from time import time

SEED_VAL = 313
# CV_VAL = 5
# HOLDOUT_SIZE = 0.3


# not needed for these datasets
def encode_data(df, cols):
    """
    Parameters:
        dataframe: list of
        cols (array-like): list of columns to encode

    """
    # encode
    l_enc = LabelEncoder()
    transformed = l_enc.fit_transform(df[[cols]])

    oh_enc = OneHotEncoder()
    encoded = oh_enc.fit_transform(transformed)

    return encoded


def scale_data(X):
    X = X.values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    return scaled


def process_abalone():
    abalone_names = [
        'sex', 'length', 'diameter', 'height', 'whole_weight',
        'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'
        ]
    df = pd.read_csv('./abalone.csv', header=None, names=abalone_names)
    df = df.dropna()

    # transform output into classification problem
    df.loc[df['rings'] < 9, 'rings'] = 1
    df.loc[(df['rings'] >= 9) & (df['rings'] <= 10), 'rings'] = 2
    df.loc[df['rings'] > 10, 'rings'] = 3

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # encode data
    X = pd.get_dummies(X)
    X = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=X.columns)

    return X, y


def process_online_shopping():
    df = pd.read_csv('./online_shoppers_intention.csv')
    df = df.dropna()

    df = df.astype({
        'OperatingSystems': 'str',
        'Browser': 'str',
        'Region': 'str',
        'TrafficType': 'str'
    })

    X = pd.concat((df.iloc[:, :-6], df.iloc[:, -5:]), axis=1)
    y = df.iloc[:, -6]

    # combine low instance browsers into one class
    mask = y.isin(['1', '2'])
    y = y.where(mask, other='3')
    # y = y.astype(str)

    # encode data
    X = pd.get_dummies(X)
    X = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=X.columns)

    return X, y


