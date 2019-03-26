import time

beginning_time = time.time()

import sys

sys.path.append("../Code/")
from utils import performance

import os
os.environ['KMP_DUPLICATE_LIB_OK']= "TRUE"

DATA_DIR = os.path.join('..', 'Data')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def timtime(t):
    print(">", int(time.time() - t), "seconds elapsed")


# sparse might be important?

develop = False

train_X = pd.concat([pd.read_pickle(os.path.join(DATA_DIR, 'train_X_1')),
                     pd.read_pickle(os.path.join(DATA_DIR, 'train_X_2'))])
train_y = pd.read_pickle(os.path.join(DATA_DIR, 'train_y'))
valid_X = pd.read_pickle(os.path.join(DATA_DIR, 'valid_X'))
valid_y = pd.read_pickle(os.path.join(DATA_DIR, 'valid_y'))

if develop:
    train_X = train_X[0:10000]
    train_y = train_y[0:10000]
    valid_X = valid_X[0:1000]
    valid_y = valid_y[0:1000]

test_X = pd.read_pickle(os.path.join(DATA_DIR, 'test_X'))

test_budget = 625000

train_budget = (test_budget * len(train_X)) // len(test_X)

from scipy.sparse import csr_matrix

train_X = csr_matrix(train_X)
valid_X = csr_matrix(valid_X)

from xgboost import XGBRegressor

# for some reason the XGBClassifier was not working

gbdt_crt_model = XGBRegressor(max_depth=10, n_estimators=75, random_state=0,
                              max_delta_step=1, objective='binary:logistic', learning_rate=0.1,
                              scale_pos_weight=1)

gbdt_crt_model.fit(train_X, train_y.click, eval_metric="logloss", eval_set=[(valid_X, valid_y.click)],
                   verbose=True, early_stopping_rounds=7)

##### save both models
import pickle

pkl_filename = "../Models/tim_xgb_click.pkl"

with open(pkl_filename, 'wb') as file:
    pickle.dump(gbdt_crt_model, file)

##### save predicted prices and predicted probabilities
