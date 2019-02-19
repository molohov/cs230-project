
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

X_train, Y_train, classes_to_index, index_to_classes = load_devset("../../data_full_150_150/dev", "../../dev.dict")
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


# **Details of our food dataset
# - Images are of shape (300, 300, 3) (can be configured in load_devset function)
# - dev set: 505 pictures (5050 if load_devset is un-gimped)

def TestCNN(input_shape):
    """
    Implementation of the TestCNN.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(num_classes, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='TestCNN')
    
    return model

## load in inceptionv3 model
K.clear_session()

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(150, 150, 3)))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = AveragePooling2D(pool_size=(8, 8), padding='same')(x)
#x = Dropout(.4)(x)
x = Flatten()(x)
#x = Dense(3 * num_classes, activation='relu')(x)
predictions = Dense(num_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

opt = SGD(lr=.01, momentum=.9)
#opt = Adam(lr=.015, beta_1=0.9, beta_2=0.999, decay=0.05, amsgrad=False)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, epochs=5, batch_size=64)

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
# You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
# 1. Create the model by calling the function above
# 2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
# 3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`
# 4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`

# 1. Create the model
#testCNN = TestCNN((300,300,3))

# 2. Compile the model to configure the learning process.
#testCNN.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])

# 3. Train the model. Choose the number of epochs and the batch size.
#    Note that if you run `fit()` again, the `model` will continue to train with the parameters it has already learnt instead of reinitializing them.
#testCNN.fit(x=X_train, y=Y_train, epochs=1, batch_size=50)

# 4. Test/evaluate the model (for now evaluating on the training set lol)
#preds = testCNN.evaluate(x = X_test, y = Y_test)
#preds = testCNN.evaluate(x = X_train, y = Y_train)
preds = model.evaluate(x = X_train, y = Y_train)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


#img_path = 'images/my_image.jpg'
#img = image.load_img(img_path, target_size=(64, 64))
#imshow(img)
#
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
#print(testCNN.predict(x))
#

# ## 5 - Other useful functions in Keras (Optional)
# 
# Two other basic features of Keras that you'll find useful are:
# - `model.summary()`: prints the details of your layers in a table with the sizes of its inputs/outputs
# - `plot_model()`: plots your graph in a nice layout. You can even save it as ".png" using SVG() if you'd like to share it on social media ;). It is saved in "File" then "Open..." in the upper bar of the notebook.
# 
# Run the following code.
#SVG(model_to_dot(testCNN).create(prog='dot', format='svg'))

