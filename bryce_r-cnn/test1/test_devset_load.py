import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict
from itertools import chain
from shutil import copyfile
from keras.utils.np_utils import to_categorical

import json

from load_devset import *

train_set_loc   = "../../data_full_150_150/train"
train_dict_loc  = "../../train_full_150_150.dict"
dev_set_loc     = "../../data_full_150_150/dev"
dev_dict_loc    = "../../dev_full_150_150.dict"

X_train, Y_train, c2i_train, i2c_train = load_devset(train_set_loc, train_dict_loc, early_termination = early_termination)
X_dev,   Y_dev,   c2i_dev,   i2c_dev   = load_devset(dev_set_loc, dev_dict_loc)

print ("c2i_train " + str(c2i_train))
print ("i2c_train " + str(i2c_train))
print ("c2i_dev   " + str(c2i_dev))
print ("i2c_dev   " + str(i2c_dev))