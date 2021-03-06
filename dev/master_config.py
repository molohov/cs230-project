num_epochs = 3
height = 150
width = 150
num_channels = 3

model_save_path = 'saved_models'
restore_weights = True
restore_weights_path = model_save_path + "/weights -  2 -  0.6531.hdf5"

plot_history = False

train_set_loc = "../data_full_" + str(height) + "_" + str(width) + "/train"
dev_set_loc = "../data_full_" + str(height) + "_" + str(width) + "/dev"
train_dict_loc = "../train_full_" + str(height) + "_" + str(width) + ".dict"
dev_dict_loc = "../dev_full_" + str(height) + "_" + str(width) + ".dict"

# Parameters
params = {'dim': (height, width),
          'batch_size': 32,
          'n_classes': 101,
          'n_channels': num_channels}
#           'shuffle': True}

# model vars
freeze_base_model  = False
num_layers_frozen  = 250
learning_rate      = 0.01
momentum           = 0.6
l2_regularizer     = 0.2


partition_dict_loc = "./partition.dict"
labels_dict_loc = "./labels.dict"


aug_params = {
    'theta': 0,
    'tx': 0,
    'ty': 0,
    'shear': 0,
    'zx': 1,
    'zy': 1,
    'mirror': True
}
