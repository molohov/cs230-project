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

import json

all_images = defaultdict(list)
train = defaultdict(list)
dev = defaultdict(list)
test = defaultdict(list)

folders = [i for i in os.listdir('../food-101/images')]

print(folders)

for f in folders:
    files = [i for i in os.listdir('../food-101/images/' + f)]
    for h in files:
        all_images[f].append(h)

for food in all_images:
    #print(food + ": " + str(len(all_images[food])))
    # just blindly separate 5/5/90 into dev/test/train
    dev[food] = all_images[food][0:50]
    test[food] = all_images[food][50:100]
    train[food] = all_images[food][100:1000]
    #print(food + "_dev: " + str(len(dev[food])))
    #print(food + "_test: " + str(len(test[food])))
    #print(food + "_train: " + str(len(train[food])))

string = json.dumps(dev)
f = open("dev.dict", "w")
f.write(string)
f.close()

string = json.dumps(test)
f = open("test.dict", "w")
f.write(string)
f.close()

string = json.dumps(train)
f = open("train.dict", "w")
f.write(string)
f.close()
