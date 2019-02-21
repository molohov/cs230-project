# coding: utf-8
import numpy as np
import keras.backend as K
from keras import layers
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Conv2D
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import layer_utils, plot_model
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
import pydot
from IPython.display import SVG
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from load_dataset import *
import math
import h5py

# >>>>> Global Vars <<<<<

# early_termination: sets the size (per food category) of the training set in pictures.  Set to -1 to use entire training set.
early_termination = 450

# epoch_count: how many epochs to run training on
epoch_count = 30

# minibatch_size: sets the minibatch size
minibatch_size = 128
length = 150
width = 150

# Data set vars
train_set_loc   = "../data_full_"+str(length)+"_"+str(width)+"/train"
train_dict_loc  = "../train_full_"+str(length)+"_"+str(width)+".dict"
dev_set_loc     = "../data_full_"+str(length)+"_"+str(width)+"/dev"
dev_dict_loc    = "../dev_full_"+str(length)+"_"+str(width)+".dict"

# model vars
freeze_base_model  = False
learning_rate      = 0.01
momentum           = 0.8
l2_regularizer     = 0.2

# output file name
output_file_name   = 'train_full.csv'

# >>>>> Subroutines <<<<<

# printAndWrite
#
# Takes a filehandle and a message as arguments.
# Prints the message to the filehandle and STDOUT
def printAndWrite(filehandle, message):
    filehandle.write(message + '\n')
    print (message)

# init
#
# Initialize the program, including loading the dataset and load the base model
def init():
    K.set_image_data_format('channels_last')

    X_train, Y_train, _, _ = load_dataset(train_set_loc, train_dict_loc, early_termination=early_termination)
    X_dev, Y_dev, _, _ = load_dataset(dev_set_loc, dev_dict_loc)

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
    x = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_regularizer), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# train_and_eval
#
# train the model on train set and evaluate it on dev set
def train_and_eval(model, X_train, Y_train, X_dev, Y_dev, output_file_name = 'train_full.csv'):
    f = open(output_file_name, 'w')
    printAndWrite (f, "Early Termination: " + str(early_termination) + " - " + str(epoch_count) + " Epochs - MiniBatch Size " + str(minibatch_size) + ' - Learning Rate ' + str(learning_rate) + ' - Momentum ' + str(momentum) + ' - L2_regularizator ' + str(l2_regularizer) + ' \n')
    printAndWrite (f, 'Epoch, Train-Loss, Train-Accuracy, Dev-Loss, Dev-Accuracy')

    for current_epoch in range(epoch_count):
        print ('Starting Epoch ' + str(current_epoch) + '/' + str(epoch_count))

        model.fit(x=X_train, y=Y_train, epochs=1, batch_size=minibatch_size)

        train_preds = model.evaluate(x = X_train, y = Y_train)
        dev_preds   = model.evaluate(x = X_dev, y = Y_dev)

        printAndWrite (f, str(current_epoch + 1) + ", " + str(train_preds[0]) + ", " + str(train_preds[1]) + ", " + str(dev_preds[0]) + ", " + str(dev_preds[1]))

    f.close()

# >>>>> Main Executable Logic <<<<<
X_train, Y_train, X_dev, Y_dev, num_classes = init()

model = create_model(num_classes, learning_rate = learning_rate, momentum = momentum, l2_regularizer = l2_regularizer)

train_and_eval(model, X_train, Y_train, X_dev, Y_dev, output_file_name = output_file_name)