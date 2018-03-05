import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

from dataset import load_imageset

i = 3
train_x, train_t, test_x, test_t = load_imageset.load_hdf5_file(i*1000, (i+1)*1000, normalize = True, validate = False)

class AlexNet:
    '''Input layer'''
    network = input_data(shape = [None, 1024, 1024, 1])
    
    '''Convolutional layer 1'''
    network = conv_2d(network, 96, 11, strides = 4, activation = 'relu')
    network = max_pool_2d(network, 3, strides = 2)
    network = batch_normalization(network)
    
    '''Convolutional layer 2'''
    network = conv_2d(network, 256, 5, strides = 1, activation = 'relu')
    network = max_pool_2d(network, 3, strides = 2)
    network = batch_normalization(network)
    
    '''Convolutional layer 3, 4'''
    network = conv_2d(network, 384, 3, strides = 1, activation = 'relu')
    network = conv_2d(network, 384, 3, strides = 1, activation = 'relu')
    
    '''Convolutional layer 5'''
    network = conv_2d(network, 256, 3, strides = 1, activation = 'relu')
    network = max_pool_2d(network, 3, strides = 2)
    network = batch_normalization(network)
    
    '''Fully connected layer 1'''
    network = fully_connected(network, 2048, activation = 'tanh')
    network = dropout(network, keep_prob = 0.5)
    
    '''Fully connected layer 2'''
    network = fully_connected(network, 2048, activation = 'tanh')
    network = dropout(network, keep_prob = 0.5)
    
    '''Fully connected layer 3'''
    network = fully_connected(network, 15, activation = 'sigmoid')
    
    
    '''Setting hyperparamters and algorithm'''
    network = regression(network, 
                         optimizer = 'Adam',
                         loss = 'binary_crossentropy',
                         learning_rate = 0.01)
    
    
    '''Set model + Save parameters + Tensorboard'''
    model = tflearn.DNN(network,
                        checkpoint_path = 'params_alexnet_cxr',
                        max_checkpoints = 1,
                        tensorboard_verbose = 2)
    
    '''Feed the CXR dataset to the model'''
    '''10% of dataset is used for validation'''
    model.fit(train_x, train_t, 
              n_epoch = 5, 
              validation_set = (test_x, test_t), 
              show_metric = True, 
              batch_size = 16, 
              snapshot_epoch = False, 
              snapshot_step = 200, 
              run_id = 'alexnet_cxr')