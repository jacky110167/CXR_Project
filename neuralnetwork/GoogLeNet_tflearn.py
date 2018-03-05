import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression


from dataset import load_imageset
train_x, train_t, test_x, test_t = load_imageset.load_hdf5_file(0, 1000, normalize = True, validate = False)


class GoogLeNet:
    network = input_data(shape = [None, 1024, 1024, 1])
    
    conv1_5_5        = conv_2d(network         , 64 , filter_size=5, strides=2, activation='relu', padding='SAME')
    pool1_3_3        = max_pool_2d(conv1_5_5, kernel_size=3, strides=2)
    pool1_3_3        = batch_normalization(pool1_3_3)
    
    conv2_3_3_reduce = conv_2d(pool1_3_3       , 64 , filter_size=1, activation='relu', padding='SAME')
    conv2_3_3        = conv_2d(conv2_3_3_reduce, 192, filter_size=3, activation='relu', padding='SAME')
    conv2_3_3        = batch_normalization(conv2_3_3)
    pool2_3_3        = max_pool_2d(conv2_3_3, kernel_size=3, strides=2)
    
    # 3a
    inception_3a_1_1        = conv_2d(pool2_3_3              , 64,  filter_size=1, activation='relu', padding='SAME')
    inception_3a_3_3_reduce = conv_2d(pool2_3_3              , 96,  filter_size=1, activation='relu', padding='SAME')
    inception_3a_3_3        = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu', padding='SAME')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3              , 16 , filter_size=1, activation='relu', padding='SAME')
    inception_3a_5_5        = conv_2d(inception_3a_5_5_reduce, 32 , filter_size=5, activation='relu', padding='SAME')
    inception_3a_pool       = max_pool_2d(pool2_3_3, kernel_size=3, strides=1)
    inception_3a_pool_1_1   = conv_2d(inception_3a_pool      , 32 , filter_size=1, activation='relu')
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)
    
    # 3b
    inception_3b_1_1        = conv_2d(inception_3a_output    , 128, filter_size=1, activation='relu', padding='SAME')
    inception_3b_3_3_reduce = conv_2d(inception_3a_output    , 128, filter_size=1, activation='relu', padding='SAME')
    inception_3b_3_3        = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', padding='SAME')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output    , 32 , filter_size=1, activation='relu', padding='SAME')
    inception_3b_5_5        = conv_2d(inception_3b_5_5_reduce, 96 , filter_size=5, activation='relu', padding='SAME')
    inception_3b_pool       = max_pool_2d(inception_3a_output, kernel_size=3, strides=1)
    inception_3b_pool_1_1   = conv_2d(inception_3b_pool      , 64 , filter_size=1, activation='relu')
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3)
    
    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2)
    
    # 4a
    inception_4a_1_1        = conv_2d(pool3_3_3              , 192, filter_size=1, activation='relu', padding='SAME')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3              , 96 , filter_size=1, activation='relu', padding='SAME')
    inception_4a_3_3        = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu', padding='SAME')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3              , 16 , filter_size=1, activation='relu', padding='SAME')
    inception_4a_5_5        = conv_2d(inception_4a_5_5_reduce, 48 , filter_size=5, activation='relu', padding='SAME')
    inception_4a_pool       = max_pool_2d(pool3_3_3, kernel_size=3, strides=1)
    inception_4a_pool_1_1   = conv_2d(inception_4a_pool      , 64 , filter_size=1, activation='relu')
    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3)
    
    # 4b
    inception_4b_1_1        = conv_2d(inception_4a_output    , 160, filter_size=1, activation='relu', padding='SAME')
    inception_4b_3_3_reduce = conv_2d(inception_4a_output    , 112, filter_size=1, activation='relu', padding='SAME')
    inception_4b_3_3        = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', padding='SAME')
    inception_4b_5_5_reduce = conv_2d(inception_4a_output    , 24 , filter_size=1, activation='relu', padding='SAME')
    inception_4b_5_5        = conv_2d(inception_4b_5_5_reduce, 64 , filter_size=5, activation='relu', padding='SAME')
    inception_4b_pool       = max_pool_2d(inception_4a_output, kernel_size=3, strides=1)
    inception_4b_pool_1_1   = conv_2d(inception_4b_pool      , 64 , filter_size=1, activation='relu')
    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3)
    
    # 4c
    inception_4c_1_1        = conv_2d(inception_4b_output    , 128, filter_size=1, activation='relu', padding='SAME')
    inception_4c_3_3_reduce = conv_2d(inception_4b_output    , 128, filter_size=1, activation='relu', padding='SAME')
    inception_4c_3_3        = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu', padding='SAME')
    inception_4c_5_5_reduce = conv_2d(inception_4b_output    , 24 , filter_size=1, activation='relu', padding='SAME')
    inception_4c_5_5        = conv_2d(inception_4c_5_5_reduce, 64 , filter_size=5, activation='relu', padding='SAME')
    inception_4c_pool       = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool        , 64 , filter_size=1, activation='relu')
    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3)
    
    # 4d
    inception_4d_1_1        = conv_2d(inception_4c_output    , 112, filter_size=1, activation='relu', padding='SAME')
    inception_4d_3_3_reduce = conv_2d(inception_4c_output    , 144, filter_size=1, activation='relu', padding='SAME')
    inception_4d_3_3        = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', padding='SAME')
    inception_4d_5_5_reduce = conv_2d(inception_4c_output    , 32 , filter_size=1, activation='relu', padding='SAME')
    inception_4d_5_5        = conv_2d(inception_4d_5_5_reduce, 64 , filter_size=5, activation='relu', padding='SAME')
    inception_4d_pool       = max_pool_2d(inception_4c_output, kernel_size=3, strides=1)
    inception_4d_pool_1_1   = conv_2d(inception_4d_pool      , 64 , filter_size=1, activation='relu')
    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3)
    
    # 4e
    inception_4e_1_1        = conv_2d(inception_4d_output    , 256, filter_size=1, activation='relu', padding='SAME')
    inception_4e_3_3_reduce = conv_2d(inception_4d_output    , 160, filter_size=1, activation='relu', padding='SAME')
    inception_4e_3_3        = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', padding='SAME')
    inception_4e_5_5_reduce = conv_2d(inception_4d_output    , 32 , filter_size=1, activation='relu', padding='SAME')
    inception_4e_5_5        = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu', padding='SAME')
    inception_4e_pool       = max_pool_2d(inception_4d_output, kernel_size=3, strides=1)
    inception_4e_pool_1_1   = conv_2d(inception_4e_pool      , 128, filter_size=1, activation='relu')
    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], mode='concat', axis=3)
    
    pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2)
    
    # 5a
    inception_5a_1_1        = conv_2d(pool4_3_3              , 256, filter_size=1, activation='relu', padding='SAME')
    inception_5a_3_3_reduce = conv_2d(pool4_3_3              , 160, filter_size=1, activation='relu', padding='SAME')
    inception_5a_3_3        = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', padding='SAME')
    inception_5a_5_5_reduce = conv_2d(pool4_3_3              , 32 , filter_size=1, activation='relu', padding='SAME')
    inception_5a_5_5        = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu', padding='SAME')
    inception_5a_pool       = max_pool_2d(pool4_3_3, kernel_size=3, strides=1)
    inception_5a_pool_1_1   = conv_2d(inception_5a_pool      , 128, filter_size=1, activation='relu')
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3, mode='concat')
    
    # 5b
    inception_5b_1_1        = conv_2d(inception_5a_output    , 384, filter_size=1, activation='relu', padding='SAME')
    inception_5b_3_3_reduce = conv_2d(inception_5a_output    , 192, filter_size=1, activation='relu', padding='SAME')
    inception_5b_3_3        = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu', padding='SAME')
    inception_5b_5_5_reduce = conv_2d(inception_5a_output    , 48 , filter_size=1, activation='relu', padding='SAME')
    inception_5b_5_5        = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', padding='SAME')
    inception_5b_pool       = max_pool_2d(inception_5a_output, kernel_size=3, strides=1)
    inception_5b_pool_1_1   = conv_2d(inception_5b_pool      , 128, filter_size=1, activation='relu')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')
    
    pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.5)

    
    '''Output layer'''
    output = fully_connected(pool5_7_7, 15, activation = 'sigmoid')
    
    network = regression(output, 
                         optimizer = 'momentum',
                         loss = 'binary_crossentropy',
                         learning_rate = 0.01)
    
    '''Set model + Save parameters + Tensorboard'''
    model = tflearn.DNN(network,
                        checkpoint_path = 'params_googlenet_cxr',
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
              run_id = 'googlenet_cxr')