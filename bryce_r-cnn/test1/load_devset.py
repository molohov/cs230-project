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

# Load dataset images and resize to square image
def load_images(root, class_to_index, early_termination = -1):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        ## FIXME forcefully load just 5 images for now to speed things up
        imgs = imgs[0:early_termination]
        class_ix = class_to_index[subdir]
        print(i, class_ix, subdir)
        for img_name in imgs:
            ## TODO: imread will be deprecated. python suggests using Image.resize, but too lazy to fix
            img_arr_rs = img.imread(join(root, subdir, img_name))
            try:
                #img_arr_rs = imresize(img_arr_rs, (size, size))
                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
            except:
                print('Skipping bad image: ', subdir, img_name)
                invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)


def load_devset(path_to_devset, path_to_dev_dict, early_termination = -1): 
    #path_to_dataset = "../food-101/images"
    #path_to_devset = "./data/dev"
    #path_to_dev_dict = "dev.dict"

    with open(path_to_dev_dict) as dev_dict:
        dev_set = json.load(dev_dict)

    class_to_index = dict(zip(dev_set, range(len(dev_set))))
    index_to_class = dict(zip(range(len(dev_set)), dev_set))
    class_to_index = {v: k for k, v in index_to_class.items()}
    sorted_class_to_index = collections.OrderedDict(sorted(class_to_index.items()))

    #print(sorted_class_to_index)

        
    X_test, y_test = load_images(path_to_devset, class_to_index, early_termination)

    # normalize. for some reason this takes an ENORMOUS amount of memory, hence commenting out for now
    #X_test = X_test / 255.

    #print('X_test shape', X_test.shape)
    #print('y_test shape', y_test.shape)


    n_classes = 101
    y_test_1hot = to_categorical(y_test, num_classes=n_classes)

    #print('y_test_1hot shape', y_test_1hot.shape)
    return X_test, y_test_1hot, class_to_index, index_to_class
