import sys

sys.path.append('../Code/')
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from utils import new_performance
import time

beginning_time = time.time()
import os

DATA_DIR = os.path.join('../', 'Data')
import pandas as pd

np.set_printoptions(precision=3, suppress=1)
plt.style.use('seaborn-notebook')


def timtime(t):
    print(">", int(time.time() - t), "seconds elapsed")


# implement budget on the agent, initalise to 6250*1000

# @title FeatureGrid
class FeatureGrid(object):

    def __init__(self, feats, true_y, impression_values, verbose=False, discount=1.0, update_thresh=1e5,
                 budget=6250 * 1000):

        self._start_state = (0)
        self._state = self._start_state
        self._number_of_states = feats.shape[0]
        self._discount = discount
        self.features = feats
        self.num_feats = feats.shape[1]
        self.true_y = true_y
        self.lambd_0 = (1.0 / 75275.275275)
        self.lambd = (1.0 / 75275.275275)
        self.impression_values = impression_values
        self.fullbudget = budget
        self.budget = budget
        self.verbose = verbose
        self.all_actions = [-0.5, -0.15, -0.08, 0, 0.08, 0.15, 0.5]
        self.update_thresh = update_thresh
        self.totalreward = 0
        self.totalr_regularizer = 159

    def reset_budget(self):
        self.budget = self.fullbudget

    def step(self, action_int):

        # need to return reward,discount,nextstate, won_bid
        action = self.all_actions[int(action_int)]
        #         if (self._state % self.update_thresh)==0: maybe do not need this to control lambda

        self.lambd = (self.lambd_0 + self.lambd_0 * action)  # lambda adjustment

        bid = (self.impression_values[self._state] / self.lambd)
        pay = self.true_y.payprice[self._state]

        if (bid <= self.budget) and (bid > 0):

            if pay < bid:

                won_bid = 1

            elif pay == bid:  # if bid=bidprice then we pick randomly

                won_bid = np.random.randint(2)

            else:

                won_bid = 0

            if won_bid:
                self.budget = self.budget - pay  # update budget consumption feature
                r = self.impression_values[self._state]

            else:
                r = 0

        else:
            r = 0
            won_bid = 0

        self.features[self._state + 1, -3] = (self.budget / self.fullbudget)  # update budget left feature
        self.totalreward += (r * 1.0 / self.totalr_regularizer * 1.0)  # update total reward/ total achievable reward
        self.features[self._state + 1, -2] = self.totalreward * 1.0  # update total reward

        self.features[self._state + 1, -1] = (
                                                     self._number_of_states * 1.0 - self._state * 1.0) / self._number_of_states * 1.0  # total time left ratio

        #         if (self._state%100000)==0:
        #             print("budget left",self.features[self._state+1,-3])
        #             print("total r",self.totalreward)
        #             print("time left",self.features[self._state+1,-1])
        #             print("LAMBDA IS: ",self.lambd,"  ACTION WAS: ",action)

        next_s = self.features[self._state + 1, :]
        self._state += 1
        discount = 1
        if self.verbose:
            print("your bid was:  ", bid, " and won_bid is =", won_bid)
            print("you paid: ", pay)
            print("state is: ", self._state)
            print("budget is: ", self.budget)
            print('total reward is', self.totalreward)

        clicks = self.true_y.click[self._state] * won_bid  # if you got a click or not

        return r, discount, next_s, won_bid, clicks

    def get_obs(self):

        return self.features[self._state, :]

    def int_to_features(self, int_state):
        return self.features[int_state, :]

    def number_of_features(self):
        return self.num_feats

    def number_of_actions(self):
        return 7


if __name__ == "__main__":

    train_X = np.load('./train_X.npy')
    train_y = np.load('./train_y.pkl')
    test = False
    model = False

    if test:
        # Test you can retrieve features

        feats = np.ones((10, 10))
        lables = np.ones((10,))
        impression_vals = np.linspace(0.1, 0.5, 10)
        # Instantiate the non tabular version of the environment.
        feat_grid = FeatureGrid(feats, lables, impression_vals)
        print("get features is:", feat_grid.int_to_features(1))
        print("get number of features is:", feat_grid.number_of_features())
        print("get number of actions is:", feat_grid.number_of_actions())
        pass

    # advanced testing - step function
    if model:
        import pickle
        pf = "../Models/tim_xgb_click.pkl"
        with open(pf, 'rb') as file:
            GBDT = pickle.load(file)
        train_X2 = np.load('./train_X2.npy')
        impression_values = GBDT.predict(train_X2)
    else:
        impression_values = np.linspace(0.1, 0.5, 10) # TANYAS temp fix to emulate the result from XGBOOST

    feat_grid = FeatureGrid(train_X, train_y, impression_values, verbose=True)
    print("######################")
    print("###########TEST1###########")
    print("step output is:", len(feat_grid.step(-0.5)))
    print("######################")
    print("###########TEST2###########")
    print("reward output is:", feat_grid.step(-0.5)[0])
    print("######################")
    print("###########TEST3###########")
    print("discount output is:", feat_grid.step(-0.5)[1])
    print("######################")
    print("###########TEST4###########")

    print("next_s output is:", feat_grid.step(-0.5)[2].shape)

    pass
