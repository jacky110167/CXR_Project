import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization


from dataset import load_imageset
train_x, train_t, test_x, test_t = load_imageset.load_hdf5_file(0, 1000, normalize = True, validate = False)


class GoogLeNet:
    network = input_data(shape = [None, 1024, 1024, 1])
    
    network = conv_2d(network, 64, 9, strides = 4, activation = 'relu', bias = False)
    
    '''Bottleneck'''
    network = tflearn.residual_bottleneck(network, nb_blocks = 3, bottleneck_size = 16, out_channels = 64)
    network = tflearn.residual_bottleneck(network, nb_blocks = 1, bottleneck_size = 32, out_channels = 128, downsample = True)
    network = tflearn.residual_bottleneck(network, nb_blocks = 2, bottleneck_size = 32, out_channels = 128)
    network = tflearn.residual_bottleneck(network, nb_blocks = 1, bottleneck_size = 64, out_channels = 256, downsample = True)
    network = tflearn.residual_bottleneck(network, nb_blocks = 2, bottleneck_size = 64, out_channels = 256)
    network = batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = global_avg_pool(network)

    '''Output layer'''
    output = fully_connected(network, 15, activation = 'sigmoid')
    
    network = regression(output, 
                         optimizer = 'momentum',
                         loss = 'binary_crossentropy',
                         learning_rate = 0.01)
    
    '''Set model + Save parameters + Tensorboard'''
    model = tflearn.DNN(network,
                        checkpoint_path = 'params_resnet_cxr',
                        max_checkpoints = 1,
                        tensorboard_verbose = 0)
    
    '''Feed the oxflowers17 dataset to the model'''
    model.fit(train_x, train_t, 
              n_epoch = 10, 
              validation_set = (test_x, test_t), 
              show_metric = True, 
              batch_size = 16, 
              snapshot_epoch = False, 
              snapshot_step = 100,
              run_id = 'resnet_cxr')