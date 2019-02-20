import numpy as np
import os
from os import listdir
from os.path import join
import shutil
import collections
from collections import defaultdict
from itertools import chain
from keras.utils.np_utils import to_categorical

import json
from PIL import Image

# Load from a folder and resize to square image
def load_images(root, class_to_index, early_termination = -1):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    for i, subdir in enumerate(listdir(root)):
        #load images in the directory
        imgs = listdir(join(root, subdir))
        #load in however many images are defined by early_termination
        imgs = imgs[0:early_termination]
        # grab the index of the class from class_to_index
        class_ix = class_to_index[subdir]
        print(i, class_ix, subdir)
        for img_name in imgs:
            img_arr = np.array(Image.open(join(root, subdir, img_name)))
            try:
                all_imgs.append(img_arr)
                all_classes.append(class_ix)
            except:
                print('Skipping bad image: ', subdir, img_name)
                invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)


# Load images and labels into array and return them
def load_dataset(path_to_dataset, path_to_dict, early_termination = -1): 
    with open(path_to_dict) as data_dict:
        dataset = json.load(data_dict)

    num_classes = len(dataset)

    # Create an index for each class, which are passed into the load_images function
    class_to_index = dict(zip(dataset, range(num_classes)))
    index_to_class = dict(zip(range(num_classes), dataset))
    class_to_index = {v: k for k, v in index_to_class.items()}
    sorted_class_to_index = collections.OrderedDict(sorted(class_to_index.items()))

    X_test, Y_test = load_images(path_to_dataset, class_to_index, early_termination)

    # normalize. for some reason this takes an ENORMOUS amount of memory, hence commenting out for now
    #X_test = X_test / 255.

    Y_test_1hot = to_categorical(Y_test, num_classes=num_classes)

    return X_test, Y_test_1hot
