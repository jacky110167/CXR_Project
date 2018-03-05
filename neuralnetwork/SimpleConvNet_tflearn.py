import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

from dataset import load_imageset

i = 0
train_x, train_t, test_x, test_t = load_imageset.load_hdf5_file(i*1000, (i+1)*1000, normalize = True, validate = False)

class SimpleConvNet_tflearn:
    
    network = input_data(shape = [None, 1024, 1024, 1])
    
    network = conv_2d(network, 128,  9, strides = 4, activation = 'relu', padding='SAME')   
    network = max_pool_2d(network, 4, strides = 2)
    network = batch_normalization(network)   
    
    network = conv_2d(network, 512, 3, strides = 1, activation = 'relu', padding='SAME') 
    network = max_pool_2d(network, 4, strides = 2) 
    network = batch_normalization(network)  
    
    network = conv_2d(network, 1024, 3, strides = 1, activation = 'relu', padding='SAME')
    
    '''Output layer'''
    network = fully_connected(network, 15, activation = 'sigmoid')
    
    
    '''Setting hyperparamters and algorithm'''
    network = regression(network, 
                         optimizer = 'Adam',
                         loss = 'binary_crossentropy',
                         learning_rate = 0.05)
    
    
    '''Set model + Save parameters + Tensorboard'''
    model = tflearn.DNN(network,
                        checkpoint_path = 'params_fcn_cxr',
                        max_checkpoints = 1,
                        tensorboard_verbose = 2)
    
    '''Feed the CXR dataset to the model'''
    '''10% of dataset is used for validation'''
    model.fit(train_x, train_t, 
              n_epoch = 10, 
              validation_set = (test_x, test_t), 
              show_metric = True, 
              batch_size = 16, 
              snapshot_epoch = False, 
              snapshot_step = 300, 
              run_id = 'fcn_cxr')