import os
import numpy as np
from numpy import ndarray
import h5py
import json
from feature_extract import *

def load_data():
    used_train, used_test = feat_extract()
    X_s = h5py.File('X_pos.h5', 'r')
    X_train = X_s['X_train'][:]
    X_test = X_s['X_test'][:]
    X_s.close()
    init = False
    Y_train = None
    Y_test = None
    ey_train = json.load(open("ey_train_pos.json", 'r'))
    ey_test = json.load(open("ey_test_pos.json", 'r'))
    for vid in used_train:
        if not init:
            Y_train = np.asarray(ey_train[vid]).reshape(1, np.asarray(ey_train[vid]).shape[0])
            init = True
        else:
            Y_train = np.concatenate((Y_train, np.asarray(ey_train[vid]).reshape(1, np.asarray(ey_train[vid]).shape[0])), axis = 0)
    init = False
    for vid in used_test:
        if not init:
            Y_test = np.asarray(ey_test[vid]).reshape(1, np.asarray(ey_test[vid]).shape[0])
            init = True
        else:
            Y_test = np.concatenate((Y_test, np.asarray(ey_test[vid]).reshape(1, np.asarray(ey_test[vid]).shape[0])), axis = 0)
    print(type(X_train), type(X_test), type(Y_train), type(Y_test))
    print (X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    return (X_train, Y_train, X_test, Y_test)
