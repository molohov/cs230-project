
import numpy as np
import keras
from PIL import Image
from os.path import join
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import random
import tensorflow as tf
import master_config
import sys
import matplotlib.pyplot as plt

aug_params = {}
aug_params['theta'] = [0, 90, 180, 270]
aug_params['shear'] = [0, 10, 20, 30]
aug_params['zx'] = [1, 0.5, 2]
aug_params['zy'] = aug_params['zx']
aug_params['mirror'] = [True, False]

img = mpimg.imread(sys.argv[1])
original_image = img


for theta in aug_params['theta']:
    for shear in aug_params['shear']:
        for z in aug_params['zx']:
            for mirror in aug_params['mirror']:
                img = tf.keras.preprocessing.image.apply_affine_transform(
                    img,
                    theta=theta,
                    shear=shear,
                    zx=z,
                    zy=z,
                    row_axis=0,
                    col_axis=1,
                    channel_axis=2,
                    fill_mode='nearest',
                )
                if mirror:
                    img = np.fliplr(img)
                imgplot = plt.imshow(img)
                plt.savefig("./augmented/" + str('theta_') + str(theta) + str('shear_') + str(shear) + str('z_') +str(z) +str('mirror_') + str(mirror) +str(".jpg"))
                img = original_image
print(count)

