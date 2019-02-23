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
from DataGenerator import DataGenerator
import math
import h5py
# import parallelTestModule
import json

num_epochs = 10

height = 150
width = 150
num_channels = 3
train_set_loc = "../data_full_" + str(height) + "_" + str(width) + "/train"
dev_set_loc = "../data_full_" + str(height) + "_" + str(width) + "/dev"

# Parameters
params = {'dim': (height,width),
          'batch_size': 64,
          'n_classes': 101,
          'n_channels': num_channels,
          'shuffle': True}

# model vars
freeze_base_model  = False
learning_rate      = 0.01
momentum           = 0.8
l2_regularizer     = 0.2


partition_dict_loc = "./partition.dict"
labels_dict_loc = "./labels.dict"

with open(partition_dict_loc,'r') as inf:
    partition_dict = eval(inf.read())

with open(labels_dict_loc,'r') as inf:
    labels_dict = eval(inf.read())

partition = partition_dict
labels = labels_dict

# create_model
#
# Create the neural network model
def create_model(num_classes=params['n_classes'], learning_rate = 0.01, momentum = 0.8, l2_regularizer = 0.05):
    ## load in inceptionv3 model
    K.clear_session()
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(height, width, num_channels)), pooling='max')

    if freeze_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    # Custom layers after base model's output
    x = base_model.output
    #x = AveragePooling2D(pool_size=(4, 4), padding = 'same')(x)
    #x = Dropout(.4)(x)
    #x = Flatten()(x)
    x = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_regularizer), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# # Datasets
# partition = # IDs
# labels = # Labels

# Generators
def main():
    training_generator = DataGenerator(partition['train'], labels, train_set_loc, 'train', **params)
    validation_generator = DataGenerator(partition['validation'], labels, dev_set_loc, 'dev', **params)

    # Design model
    model = create_model(params['n_classes'], learning_rate = learning_rate, momentum = momentum, l2_regularizer = l2_regularizer)

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        epochs=num_epochs)

# if __name__ == '__main__':
#     extractor = parallelTestModule.ParallelExtractor()
#     extractor.runInParallel(numProcesses=2, numThreads=4)

if __name__ == "__main__":
    main()

