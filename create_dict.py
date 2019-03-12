import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

from os import listdir, makedirs
from os.path import isdir
import stat
import collections
from collections import defaultdict
from itertools import chain

import json

train     = defaultdict(list)
dev       = defaultdict(list)
test      = defaultdict(list)

length = 150
width = 150

train_dir        =  "data_full_"+str(length)+"_"+str(width)+"/train"
train_dict       = "train_full_"+str(length)+"_"+str(width)+".dict"
dev_dir          =  "data_full_"+str(length)+"_"+str(width)+"/dev"
dev_dict         = "dev_full_"+str(length)+"_"+str(width)+".dict"
test_dir         =  "data_full_"+str(length)+"_"+str(width)+"/test"
test_dict        = "test_full_"+str(length)+"_"+str(width)+".dict"

# update dict

for food in sorted(listdir(train_dir)):
    dest_dir = train_dir + "/" + food + "/"
    files = sorted(listdir(dest_dir))
    for h in files:
        train[food].append(h)
        
string = json.dumps(train)
f = open(train_dict, "w")
f.write(string)
f.close()

for food in sorted(listdir(dev_dir)):
    dest_dir = dev_dir + "/" + food + "/"
    files = sorted(listdir(dest_dir))
    for h in files:
        dev[food].append(h)
        
string = json.dumps(dev)
f = open(dev_dict, "w")
f.write(string)
f.close()

for food in sorted(listdir(test_dir)):
    dest_dir = train_dir + "/" + food + "/"
    files = sorted(listdir(dest_dir))
    for h in files:
        test[food].append(h)

string = json.dumps(test)
f = open(test_dict, "w")
f.write(string)
f.close()