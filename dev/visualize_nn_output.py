from keras import backend as K
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
import numpy as np
from vis.visualization import visualize_activation
from vis.visualization import get_num_filters
from vis.utils import utils
from keras import activations

from matplotlib import pyplot as plt

from load_weights import main
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Conv2D
from vis.input_modifiers import Jitter


## you must install this version of keras-vis from the repo in order for this to work
## pip install git+https://github.com/raghakot/keras-vis.git --upgrade

## otherwise you'll get this weird error
## tensorflow.python.framework.errors_impl.InvalidArgumentError: input_1:0 is both fed and fetched.

if __name__ == '__main__':

    # load model
    model = main(returnModel=True)
    #model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(150, 150, 3)))
    print('Model loaded.')
    #model.summary()
    single_filter = False
    #single_filter = True

    # The name of the layer we want to visualize
    layer_name = 'dense_1'
    #layer_name = 'conv2d_1'
    layer_idx = utils.find_layer_idx(model, layer_name)

    #swap activations of last layer with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    # Visualize all filters in this layer.
    filters = None
    if (single_filter == False):
        filters = np.arange(get_num_filters(model.layers[layer_idx]))
    else:
        filters = ([0])

    # Generate input image for each filter.
    vis_images = []
    print ("Processing ", len(filters), " filters")
    for idx in filters:
        print ("Processing filter", idx, "/", len(filters))
        img = visualize_activation(model, layer_idx, filter_indices=idx, tv_weight=0., lp_norm_weight=0., max_iter=200, input_modifiers=[Jitter(16)])
        
        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Output class {}'.format(idx))    
        vis_images.append(img)

    # Generate stitched image palette with 8 cols.
    stitched = utils.stitch_images(vis_images, cols=8)    
    plt.axis('off')
    plt.title(layer_name)
    plt.imsave(layer_name + ".png", stitched)
    #plt.imshow(stitched)
    #plt.show()
    #stitched = Image.fromarray(stitched)
    #stitched.save(layer_name + '.png')
