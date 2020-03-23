import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve

from time import time

SEED_VAL = 313
HOLDOUT_SIZE = 0.3
CV_VAL = 5


# not needed for these datasets, can use get_dummies instead
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


def process_abalone(scaler='minmax', tt_split=False):
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
    if scaler == 'minmax':
        X = pd.DataFrame(MinMaxScaler().fit_transform(X.values), columns=X.columns)
    else:
        X = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=X.columns)

    if tt_split:
        return train_test_split(X, y, test_size=HOLDOUT_SIZE, random_state=SEED_VAL)
    else:
        return X, y


def process_abalone_w_clusters(X_clusters, scaler='minmax'):
    abalone_names = [
        'sex', 'length', 'diameter', 'height', 'whole_weight',
        'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'
        ]
    df = pd.read_csv('./abalone.csv', header=None, names=abalone_names)
    df = df.dropna()

    # attach cluster labels
    df['clusters'] = X_clusters
    df = df.astype({'clusters': 'str'})
    abalone_cols = [
        'sex', 'length', 'diameter', 'height', 'whole_weight',
        'shucked_weight', 'viscera_weight', 'shell_weight', 'clusters', 'rings'
        ]
    df = df[abalone_cols]

    # transform output into classification problem
    df.loc[df['rings'] < 9, 'rings'] = 1
    df.loc[(df['rings'] >= 9) & (df['rings'] <= 10), 'rings'] = 2
    df.loc[df['rings'] > 10, 'rings'] = 3

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # encode data
    X = pd.get_dummies(X)
    if scaler == 'minmax':
        X = pd.DataFrame(MinMaxScaler().fit_transform(X.values), columns=X.columns)
    else:
        X = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=X.columns)

    return train_test_split(X, y, test_size=HOLDOUT_SIZE, random_state=SEED_VAL)


def process_online_shopping(scaler='minmax', tt_split=False):
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
    if scaler == 'minmax':
        X = pd.DataFrame(MinMaxScaler().fit_transform(X.values), columns=X.columns)
    elif scaler == 'standard':
        X = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=X.columns)

    if tt_split:
        return train_test_split(X, y, test_size=HOLDOUT_SIZE, random_state=SEED_VAL)
    else:
        return X, y


def model_train_score(estimator, X_train, y_train):
    return estimator.score(X_train, y_train)


def model_test_score(estimator, X_test, y_test):
    return estimator.score(X_test, y_test)
