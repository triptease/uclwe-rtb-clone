import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys

# data directory
DATA_DIR = os.path.join('..', 'Data')

# import utils and strategies
sys.path.append("../Code/")
from utils import performance, new_performance
from strategies import constant_bidding_strategy, random_bidding_strategy


def test_constant_bidding(y, bid):
    print("Results:")
    bids = constant_bidding_strategy(len(y), bid)
    new_performance(bids, y)


def test_random_bidding(y, lower=0, upper=300, mean=78, std=59, distribution='uniform'):
    print("Results:")
    if distribution == 'uniform':
        bids = random_bidding_strategy(len(y), min_bet=lower, max_bet=upper, distribution="uniform")
    else:
        bids = random_bidding_strategy(len(y), mean=mean, std=std, distribution="normal")
    new_performance(bids, y)


def evaluate_constant_strategy(train_X, train_Y, valid_X, valid_Y, low, high, step=None, shuffle=False):
    start = time.time()
    constant_bids = np.arange(low, high, step)
    vals = pd.DataFrame(constant_bids).reset_index()
    num_clicks = np.zeros_like(constant_bids)
    num_clicks_valid = np.zeros_like(constant_bids)

    if shuffle:
        train_X, train_y = shuffle(train_X, train_Y, random_state=0)
        valid_X, valid_y = shuffle(valid_X, valid_Y, random_state=0)

    for i in range(len(constant_bids)):
        _, c, _, _, _, _, _ = new_performance(constant_bidding_strategy(len(train_X), constant_bids[i]), train_Y,
                                              verbose=False)
        _, c_v, _, _, _, _, _ = new_performance(constant_bidding_strategy(len(valid_X), constant_bids[i]), valid_Y,
                                                verbose=False)
        num_clicks[i] = c
        num_clicks_valid[i] = c_v
    vals = vals.join(pd.DataFrame({'#Clicks_Train': num_clicks}).reset_index(drop=True))
    vals = vals.join(pd.DataFrame({'#Clicks_Valid': num_clicks_valid}).reset_index(drop=True))

    print("time spent:- {0:.2f} seconds".format(time.time() - start))
    return (constant_bids[np.argmax(num_clicks)], constant_bids[np.argmax(num_clicks_valid)], vals)


def evaluate_random_strategy(train_X, train_Y, valid_X, valid_Y, low, high, step=None, randomType='uniform',
                             shuffle=False):
    start = time.time()
    constant_bids = np.arange(low, high, step)
    num_clicks = np.zeros(shape=(constant_bids.shape[0], constant_bids.shape[0]), dtype=int)
    num_clicks_valid = np.zeros(shape=(constant_bids.shape[0], constant_bids.shape[0]), dtype=int)
    params = {'lower': [constant_bids[l] for l in range(len(constant_bids)) for u in range(len(constant_bids))],
              'upper': [constant_bids[u] for l in range(len(constant_bids)) for u in range(len(constant_bids))]}
    vals = pd.DataFrame(params).reset_index(drop=True)

    if shuffle:
        train_X, train_y = shuffle(train_X, train_Y, random_state=0)
        valid_X, valid_y = shuffle(valid_X, valid_Y, random_state=0)

    for lower in range(0, len(constant_bids) - 1):
        for upper in range(lower + 1, len(constant_bids)):
            _, c, _, _, _, _, _ = new_performance(
                random_bidding_strategy(len(train_X), min_bet=constant_bids[lower], max_bet=constant_bids[upper],
                                        distribution=randomType), train_Y, verbose=False)
            _, c_v, _, _, _, _, _ = new_performance(
                random_bidding_strategy(len(valid_X), min_bet=constant_bids[lower], max_bet=constant_bids[upper],
                                        distribution=randomType), valid_Y, verbose=False)
            num_clicks[lower, upper] = c
            num_clicks_valid[lower, upper] = c_v
    vals = vals.join(pd.DataFrame({'n_clicks_train': [num_clicks[l, u] for l in range(len(constant_bids)) for u in
                                                      range(len(constant_bids))]}).reset_index(drop=True))
    vals = vals.join(pd.DataFrame({'n_clicks_valid': [num_clicks_valid[l, u] for l in range(len(constant_bids)) for u in
                                                      range(len(constant_bids))]}).reset_index(drop=True))
    best_train_lower, best_train_upper = np.unravel_index(num_clicks.argmax(), num_clicks.shape)
    best_valid_lower, best_valid_upper = np.unravel_index(num_clicks_valid.argmax(), num_clicks_valid.shape)

    print("time spent:- {0:.2f} seconds".format(time.time() - start))
    return (constant_bids[best_train_lower], constant_bids[best_train_upper], constant_bids[best_valid_lower],
            constant_bids[best_valid_upper], vals, num_clicks_valid)


############################################################
############################################################
############################################################

# load data
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
validation = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))

train_X = train.drop(['click', 'bidprice', 'payprice'], axis=1)
train_y = train[['click', 'bidprice', 'payprice']].copy()

valid_X = validation.drop(['click', 'bidprice', 'payprice'], axis=1)
valid_y = validation[['click', 'bidprice', 'payprice']].copy()

del train, validation

test_constant_bidding(train_y, 33)

# train_argmax, valid_argmax, eval_table = evaluate_constant_strategy(train_X, train_y, valid_X, valid_y, 0, 301)
