"""SVHN street view house numbers dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils.data_utils import get_file
#from keras. import backend as K
import numpy as np

import scipy as sp
import scipy.io as sio
from scipy.misc import *
from keras.utils import to_categorical
import os


def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'svhn-dataset')
    base = 'http://ufldl.stanford.edu/housenumbers/'
    files = ['train_32x32.mat', 'test_32x32.mat']

    paths = []
    for fname in files:
        paths.append(get_file(fname,
                              origin=base + fname,
                              cache_subdir=dirname))



    train_dict = sio.loadmat(paths[0])

    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0

    Y_train = to_categorical(Y_train,10)
    
    test_dict = sio.loadmat(paths[1])
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    Y_test = to_categorical(Y_test,10)
    return (X_train, Y_train), (X_test, Y_test)



