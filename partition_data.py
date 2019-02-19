import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

import os
from os import listdir, makedirs
from os.path import isfile, join
import stat
import collections
from collections import defaultdict
from itertools import chain
from PIL import Image

import json

perform_resize = 1
resized_dimensions = [150, 150]

all_images = defaultdict(list)
train = defaultdict(list)
dev = defaultdict(list)
test = defaultdict(list)

source_image_dir = "../food-101/images/"
train_dir        = "data_full/train"
dev_dir          = "data_full/dev"
test_dir         = "data_full/test"
train_dict       = "train_full.dict"
dev_dict         = "dev_full.dict"
test_dict        = "test_full.dict"

folders = [i for i in os.listdir(source_image_dir)]

print(folders)

for f in folders:
    files = [i for i in os.listdir(source_image_dir + f)]
    for h in files:
        all_images[f].append(h)

for food in all_images:
    size = len(all_images[food])

    # just blindly separate 5/5/90 into dev/test/train
    dev_set_size   = int(0.05 * size)
    test_set_size  = int(0.05 * size)
    train_set_size = size - dev_set_size - test_set_size

    dev_set_start   = 0
    dev_set_end     = dev_set_start + dev_set_size
    test_set_start  = dev_set_end + 1
    test_set_end    = test_set_start + test_set_size
    train_set_start = test_set_end + 1
    train_set_end   = train_set_start + train_set_size

    #print (food + " - S: " + str(size) + " D: " + str(dev_set_start) + ":" + str(dev_set_end) + " T: " + str(test_set_start) + ":" + str(test_set_end) + " R: " + str(train_set_start) + ":" + str(train_set_end))

    dev[food] = all_images[food][dev_set_start:dev_set_end]
    test[food] = all_images[food][test_set_start:test_set_end]
    train[food] = all_images[food][train_set_start:train_set_end]

string = json.dumps(dev)
f = open(dev_dict, "w")
f.write(string)
f.close()

string = json.dumps(test)
f = open(test_dict, "w")
f.write(string)
f.close()

string = json.dumps(train)
f = open(train_dict, "w")
f.write(string)
f.close()

for food in all_images:
    print("Processing " + food + " images...")
    source_dir = source_image_dir + food + "/"
    dest_dev_dir   = dev_dir   + "/" + food + "/"
    dest_test_dir  = test_dir  + "/" + food + "/"
    dest_train_dir = train_dir + "/" + food + "/"

    os.makedirs(dest_dev_dir)
    for pic in dev[food]:
        img = Image.open(source_dir + pic)
        if perform_resize:
            width, height = img.size
            img = img.resize(resized_dimensions)
        img.save(dest_dev_dir + pic, img.format)

    os.makedirs(dest_test_dir)
    for pic in test[food]:
        img = Image.open(source_dir + pic)
        if perform_resize:
            width, height = img.size
            img = resizeimage.resize_contain(img, resized_dimensions)
        img.save(dest_test_dir + pic, img.format)

    os.makedirs(dest_train_dir)
    for pic in train[food]:
        img = Image.open(source_dir + pic)
        if perform_resize:
            width, height = img.size
            img = resizeimage.resize_contain(img, resized_dimensions)
        img.save(dest_train_dir + pic, img.format)
