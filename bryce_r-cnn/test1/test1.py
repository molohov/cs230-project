
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

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from load_devset import *

import math
import h5py

#def load_dataset():
#    train_dataset = h5py.File('datasets/train_happy.h5', "r")
#    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
#    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
#
#    test_dataset = h5py.File('datasets/test_happy.h5', "r")
#    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
#    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
#
#    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
#    
#    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
#    
#    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

## TODO: convert devset into h5 format and load it elegantly like above, to avoid wasting computation time
X_train, Y_train, classes_to_index, index_to_classes = load_devset("../../data/dev", "../../dev.dict")

# Normalize image vectors. This will EXPLODE memory due to numpy inefficiencies
#X_train = X_train_orig/255.
#X_test = X_test_orig/255.

# Reshape
#Y_train = Y_train_orig.T
#Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
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
    X = Dense(101, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='TestCNN')
    
    return model


# You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
# 1. Create the model by calling the function above
# 2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
# 3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`
# 4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`

# 1. Create the model
testCNN = TestCNN((300,300,3))

# 2. Compile the model to configure the learning process.
testCNN.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=["accuracy"])

# 3. Train the model. Choose the number of epochs and the batch size.
#    Note that if you run `fit()` again, the `model` will continue to train with the parameters it has already learnt instead of reinitializing them.
testCNN.fit(x=X_train, y=Y_train, epochs=1, batch_size=50)

# 4. Test/evaluate the model (for now evaluating on the training set lol)
#preds = testCNN.evaluate(x = X_test, y = Y_test)
preds = testCNN.evaluate(x = X_train, y = Y_train)

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

