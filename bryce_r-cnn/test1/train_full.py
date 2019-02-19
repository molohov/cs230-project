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
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from load_devset import *

import math
import h5py

# >>>>> Global Vars <<<<<

# early_termination: sets the size (per food category) of the training set in pictures.  Set to -1 to use entire training set.
early_termination = 450

# epoch_count: how many epochs to run training on
epoch_count = 30

# minibatch_size: sets the minibatch size
minibatch_size = 64

# Data set vars
train_set_loc   = "../../data_full_150_150/train"
train_dict_loc  = "../../train_full_150_150.dict"
dev_set_loc     = "../../data_full_150_150/dev"
dev_dict_loc    = "../../dev_full_150_150.dict"

# model vars
freeze_base_model  = False
learning_rate      = 0.01
momentum           = 0.8

# output file name
output_file_name   = 'train_full.csv'

# >>>>> Subroutines <<<<<

# printAndWrite
# 
# Takes a filehandle and a message as arguments.
# Prints the message to the filehandle and STDOUT
def printAndWrite(filehandle, message):
    f.write(message + '\n')
    print (message)

# init
#
# Initialize the program, including loading the dataset and load the base model
def init():
    K.set_image_data_format('channels_last')

    X_train, Y_train, _, _ = load_devset(train_set_loc, train_dict_loc, early_termination = early_termination)
    X_dev, Y_dev, _, _ = load_devset(dev_set_loc, dev_dict_loc)

    num_classes = Y_train.shape[1]

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of classes = " + str(num_classes))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))
    
    return X_train, Y_train, X_dev, Y_dev, num_classes

# create_model
#
# Create the neural network model
def create_model(num_classes, learning_rate = 0.01, momentum = 0.8, l2_regularizer = 0.05):
    ## load in inceptionv3 model
    K.clear_session()
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(150, 150, 3)))

    if freeze_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    # Custom layers after base model's output
    x = base_model.output
    x = AveragePooling2D(pool_size=(4, 4), padding = 'same')(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    x = Dense(num_classes, init='glorot_uniform', kernel_regularizer=l2(l2_regularizer), activation='softmax')(x)

    model = Model(input=base_model.input, output=x)

    opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# train_and_eval
#
# train the model on train set and evaluate it on dev set
def train_and_eval(model, X_train, Y_train, X_dev, Y_dev, output_file_name = 'train_full.csv'):
    f = open(output_file_name, 'w')
    printAndWrite (f, "Early Termination: " + str(early_termination) + " - " + str(epoch_count) + " Epochs - MiniBatch Size " + str(minibatch_size) + '\n')
    printAndWrite (f, 'Epoch, Train-Loss, Train-Accuracy, Dev-Loss, Dev-Accuracy')

    for current_epoch in range(epoch_count):
        model.fit(x=X_train, y=Y_train, epochs=1, batch_size=minibatch_size)

        train_preds = model.evaluate(x = X_train, y = Y_train)
        dev_preds   = model.evaluate(x = X_dev, y = Y_dev)

        printAndWrite (f, str(current_epoch + 1) + ", " + str(train_preds[0]) + ", " + str(train_preds[1]) + ", " + str(dev_preds[0]) + ", " + str(dev_preds[1]))

    f.close()

# >>>>> Main Executable Logic <<<<<
X_train, Y_train, X_dev, Y_dev, num_classes = init()

model = create_model(num_classes, learning_rate = learning_rate, momentum = momentum)

train_and_eval(model, X_train, Y_train, X_dev, Y_dev, output_file_name = output_file_name)