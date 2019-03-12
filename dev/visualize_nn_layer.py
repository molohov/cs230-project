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

## you must install this version of keras-vis from the repo in order for this to work
## pip install git+https://github.com/raghakot/keras-vis.git --upgrade

## otherwise you'll get this weird error
## tensorflow.python.framework.errors_impl.InvalidArgumentError: input_1:0 is both fed and fetched.

if __name__ == '__main__':

    # load model
    model = main(returnModel=True)
    print('Model loaded.')
    #model.summary()
    single_filter = True


    # The name of the layer we want to visualize
    layer_name = 'conv2d_94'
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    # Generate input image for each filter.
    vis_images = []
    if (single_filter == False):
        print ("Processing ", len(filters), " filters")
        for idx in filters:
            print ("Processing the ", idx, "th filter")
            img = visualize_activation(model, layer_idx, filter_indices=idx)
            
            # Utility to overlay text on image.
            img = utils.draw_text(img, 'Filter {}'.format(idx))    
            vis_images.append(img)
    else: 
        ### single filter output for debug
        img = visualize_activation(model, layer_idx, filter_indices=0)
        
        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Filter {}'.format(0))    
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
