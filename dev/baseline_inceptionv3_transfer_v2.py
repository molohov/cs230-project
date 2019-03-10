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
from os import makedirs
from os.path import isdir
import pydot
from IPython.display import SVG
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from DataGenerator import DataGenerator
import math
import h5py
# import parallelTestModule
import json
import master_config
from keras.callbacks import ModelCheckpoint

with open(master_config.partition_dict_loc,'r') as inf:
    partition = eval(inf.read())

with open(master_config.labels_dict_loc,'r') as inf:
    labels = eval(inf.read())

# create_model
#
# Create the neural network model
def create_model(num_classes=master_config.params['n_classes'], learning_rate = 0.01, momentum = 0.8, l2_regularizer = 0.05):
    ## load in inceptionv3 model
    K.clear_session()
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(master_config.height, master_config.width, master_config.num_channels)), pooling='max')

    if master_config.freeze_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_regularizer), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    if master_config.restore_weights:
        save_weight_filepath = master_config.restore_weights_path
        model.load_weights(save_weight_filepath)

    opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# # Datasets
# partition = # IDs
# labels = # Labels

# Generators
def main():
    training_generator = DataGenerator(partition['train'], labels, master_config.train_set_loc, 'train', **master_config.params)
    validation_generator = DataGenerator(partition['validation'], labels, master_config.dev_set_loc, 'dev', **master_config.params)

    # Design model
    model = create_model(master_config.params['n_classes'], learning_rate = master_config.learning_rate, momentum = master_config.momentum, l2_regularizer = master_config.l2_regularizer)

    # Train model on dataset
    if isdir(master_config.model_save_path) is False:
        makedirs(master_config.model_save_path)

    save_weight_filepath = master_config.model_save_path + "/weights - {epoch: 02d} - {val_acc: .4f}.hdf5"
    checkpoint = ModelCheckpoint(save_weight_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        epochs=master_config.num_epochs,
                        callbacks=callbacks_list)

    if master_config.plot_history:
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

# if __name__ == '__main__':
#     extractor = parallelTestModule.ParallelExtractor()
#     extractor.runInParallel(numProcesses=2, numThreads=4)

if __name__ == "__main__":
    main()

