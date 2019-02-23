length = 150
width = 150
train_set_loc = "../data_full_" + str(length) + "_" + str(width) + "/train"
train_dict_loc = "../train_full_" + str(length) + "_" + str(width) + ".dict"
dev_set_loc = "../data_full_" + str(length) + "_" + str(width) + "/dev"
dev_dict_loc = "../dev_full_" + str(length) + "_" + str(width) + ".dict"

# Parameters
params = {'dim': (length, width),
          'batch_size': 32,
          'n_classes': 101,
          'n_channels': 3,
          'shuffle': True}

# model vars
freeze_base_model  = False
learning_rate      = 0.01
momentum           = 0.8
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