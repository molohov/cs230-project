import numpy as np
import keras
from PIL import Image
from os.path import join
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import random
import tensorflow as tf
import master_config


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, path_to_dataset, subdir, batch_size=32, dim=(150, 150), n_channels=3,
             n_classes=10, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.path_to_dataset = path_to_dataset
        self.subdir = subdir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        master_config.aug_params['theta'] = random.choice([0, 90, 180, 270])
        master_config.aug_params['shear'] = random.choice([0, 10, 20, 30])
        master_config.aug_params['zx'] = random.choice([1, 0.5, 2])
        master_config.aug_params['zy'] = master_config.aug_params['zx']
        master_config.aug_params['mirror'] = random.choice([True, False])

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # print(X.shape)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            # print(ID)
            # print(np.array(Image.open(self.path_to_dataset + "/" + ID)).shape)
            img = mpimg.imread(self.path_to_dataset+"/"+ID)
            img = tf.keras.preprocessing.image.apply_affine_transform(
                img,
                theta=master_config.aug_params['theta'],
                shear=master_config.aug_params['shear'],
                zx=master_config.aug_params['zx'],
                zy=master_config.aug_params['zy'],
                row_axis=0,
                col_axis=1,
                channel_axis=2,
                fill_mode='nearest',
            )
            if master_config.aug_params['mirror']:
                img = np.fliplr(img)
            X[i,] = np.array(img) / 255
            # Store class
            # print(self.labels)
            # print(ID)
            # print(self.labels[ID])
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # print(len(list_IDs_temp))

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
