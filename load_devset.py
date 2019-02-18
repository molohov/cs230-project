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


import json

path_to_dataset = "../food-101/images"
path_to_devset = "./data/dev"

with open("dev.dict") as dev_dict:
    dev_set = json.load(dev_dict)

class_to_ix = dict(zip(dev_set, range(len(dev_set))))
ix_to_class = dict(zip(range(len(dev_set)), dev_set))
class_to_ix = {v: k for k, v in ix_to_class.items()}
sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

#print(sorted_class_to_ix)

if not os.path.isdir(path_to_devset):
    os.makedirs(path_to_devset)
    for folder in dev_set:
        os.makedirs(path_to_devset + "/" + folder)
        for image in dev_set[folder]:
            copyfile(path_to_dataset + "/" + folder + "/" + image, path_to_devset + "/" + folder + "/" + image)
    
else:
    print('dev folder already exists')
