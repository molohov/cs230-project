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
from PIL import Image

import json

perform_resize = True
length = 150
width = 150
resized_dimensions = [length, width]

all_files = defaultdict(list)
train     = defaultdict(list)

train_dir        =  "data_full_"+str(length)+"_"+str(width)+"/train"
train_dict       = "train_full_"+str(length)+"_"+str(width)+".dict"

source_image_dir = "image_download/"

folders = sorted(listdir(source_image_dir))

for f in folders:
    files = sorted(listdir(source_image_dir + f))
    for h in files:
        all_files[f].append(h)

for food in all_files:
    print("Processing " + food + " images...")
    dest_train_dir = train_dir + "/" + food + "/"
    source_dir = source_image_dir + food + "/"

    if not isdir(dest_train_dir):
        makedirs(dest_train_dir)
    for pic in all_files[food]:
        try:
            img = Image.open(source_dir + pic)
            if perform_resize:
                img = img.resize(resized_dimensions)
                img.save(dest_train_dir + pic, img.format)
        except:
            # skip
            print ("Encountered error handling " + source_dir + pic)

# update dict

for food in sorted(listdir(train_dir)):
    dest_train_dir = train_dir + "/" + food + "/"
    files = sorted(listdir(dest_train_dir))
    for h in files:
        train[food].append(h)
        
string = json.dumps(train)
f = open(train_dict, "w")
f.write(string)
f.close()