import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imresize

from os import listdir, makedirs, remove
from os.path import isdir
from PIL import Image

import json

check_download = False
length = 150
width = 150
resized_dimensions = [length, width]

dirs_to_check    = []

if check_download:
    dirs_to_check    = ["image_download"]
else:
    train_dir        = "data_full_"+str(length)+"_"+str(width)+"/train"
    dev_dir          = "data_full_"+str(length)+"_"+str(width)+"/dev"
    test_dir         = "data_full_"+str(length)+"_"+str(width)+"/test"
    dirs_to_check    = [train_dir, dev_dir, test_dir]


for source_image_dir in dirs_to_check:
    for food in listdir(source_image_dir):
        source_dir = source_image_dir + '/' + food + "/"

        for pic in listdir(source_dir):
            image_path = source_dir + pic
            try:
                img = mpimg.imread(image_path)
                img_shape = img.shape
                if img_shape[0] is not 150 or img_shape[1] is not 150 or img_shape[2] is not 3:
                    print ("image ", image_path, " shape ", img.shape)
                    remove(image_path)
            except:
                print ("unable to check ", image_path)
