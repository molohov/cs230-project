# coding: utf-8
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from load_devset import *

import math
import h5py

import sys, getopt

## this code loads the dev set in manually and then converts it to h5
#X_dev, Y_dev, classes_to_index, index_to_classes = load_devset("../../data/dev", "../../dev.dict")
#with h5py.File('dev_data.h5', 'w') as file:
#    file.create_dataset("data", data=X_dev)
#    file.create_dataset("labels", data=Y_dev)

## this loads the h5 database
#dev_set = h5py.File('../../dev_data.h5')
#
#X_train = np.array(dev_set["data"])
#Y_train = np.array(dev_set["labels"])

X_train, Y_train, classes_to_index, index_to_classes = load_devset("../data_full/dev", "../dev_full.dict")
#X_dev, Y_dev, classes_to_index, index_to_classes = load_devset("../data_full/dev", "../dev_full.dict")
num_classes = Y_train.shape[1]


# Normalize image vectors. This will EXPLODE memory due to numpy inefficiencies
#X_train = X_train_orig/255.
#X_test = X_test_orig/255.

# Reshape
#Y_train = Y_train_orig.T
#Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of classes = " + str(num_classes))
#print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
#print ("X_test shape: " + str(X_test.shape))
#print ("Y_test shape: " + str(Y_test.shape))

## load in inceptionv3 model
K.clear_session()

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(150, 150, 3)))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
#x = AveragePooling2D(pool_size=(3, 3))(x)
x = Dropout(.4)(x)
x = Flatten()(x)
predictions = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, epochs=1, batch_size=32)

## code from source that we may need later
#checkpointer = ModelCheckpoint(filepath='model4.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
#csv_logger = CSVLogger('model4.log')
#
#def schedule(epoch):
#    if epoch < 15:
#        return .01
#    elif epoch < 28:
#        return .002
#    else:
#        return .0004
#lr_scheduler = LearningRateScheduler(schedule)
#
#model.fit_generator(train_generator,
#                    validation_data=test_generator,
#                    nb_val_samples=X_test.shape[0],
#                    samples_per_epoch=X_train.shape[0],
#                    nb_epoch=1,
#                    verbose=2,
#                    callbacks=[lr_scheduler, csv_logger, checkpointer])
#
preds = model.evaluate(x = X_train, y = Y_train)

print("Train Set")
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

preds = model.evaluate(x = X_dev, y = Y_dev)
print("Dev Set")
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
