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
from load_dataset import load_dataset
import math
import h5py
# import parallelTestModule
import json
import master_config
from keras.callbacks import ModelCheckpoint
from DataGenerator import DataGenerator


with open(master_config.partition_dict_loc,'r') as inf:
    partition = eval(inf.read())

with open(master_config.labels_dict_loc,'r') as inf:
    labels = eval(inf.read())

# create_model
#
# Create the neural network model
def create_model(num_classes=master_config.params['n_classes'], l2_regularizer = 0.05):
    ## load in inceptionv3 model
    K.clear_session()
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(master_config.height, master_config.width, master_config.num_channels)), pooling='max')

    if master_config.freeze_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    # Custom layers after base model's output
    x = base_model.output
    x = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_regularizer), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


# # Datasets
# partition = # IDs
# labels = # Labels

# Generators
def main(returnModel=False):
    printWeights = False
    save_weight_filepath = master_config.restore_weights_path
    validation_generator = DataGenerator(partition['validation'], labels, master_config.dev_set_loc, 'dev', **master_config.params)
    # print(validation_generator.list_IDs)
    print(len(validation_generator.list_IDs))


    # Design model
    model = create_model(master_config.params['n_classes'])
    model.load_weights(save_weight_filepath)

    if returnModel:
        return model    
 
    if printWeights:
        for layer in model.layers:
            weights = layer.get_weights() # list of numpy arrays
            print (weights)
 
    prediction = model.predict_generator(
        generator=validation_generator,
        use_multiprocessing=True,
        workers=4,
    )
 
    rng = np.arange(1, 102)

    prediction_index = np.argmax(prediction, axis=1)

    mean = np.mean(prediction, axis=0)
    list_of_images = validation_generator.list_IDs
    list_of_predicted_images = list_of_images[0:prediction.shape[0]]
    # print(list_of_predicted_images)

    with open("./labels.dict", 'r') as inf:
        dict_from_file = eval(inf.read())


    i = 0
    right = 0
    for img in list_of_predicted_images:
        if dict_from_file[img] == prediction_index[i]:
            right = right + 1
        i = i+1

    accuracy = right/i
    print("Cacluated Accuracy =", accuracy)

    print ("prediction shape = ", prediction.shape)
    print ("rng shape = ", rng.shape)
    print ("mean shape = ", mean.shape)
 
    plt.barh(rng, mean[:,])
    plt.title('categorical accuracy')
    plt.ylabel('category')
    plt.xlabel('mean confidence')
    #plt.legend(['validation'], loc='upper left')
    # plt.show()

if __name__ == "__main__":
    main()

