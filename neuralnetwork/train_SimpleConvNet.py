import numpy as np
import matplotlib.pyplot as plt
from dataset import load_imageset
from neuralnetwork import SimpleConvNet
from commonutil import trainer


train_images, train_labels, test_images, test_labels = load_imageset.load_hdf5_file(200, normalize = True, validate = False)

train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
test_images  = test_images.reshape( test_images.shape[0] , 1, test_images.shape[1] , test_images.shape[2] )

print('Shape of train images:', train_images.shape)
print('Shape of test images: ', test_images.shape)
print('Shape of train labels:', train_labels.shape)
print('Shape of test labels: ', test_labels.shape)

max_epochs = 20

network = SimpleConvNet.SimpleConvNet(input_dim = (1, 1024, 1024),
                                      conv_params = {
                                                    'filter_num' : (64 , 128, 128, 256),
                                                    'filter_size': (13 ,   7,   5,   3),
                                                    'stride'     : ( 3 ,   2,   1,   1),
                                                    },
                                      hidden_size = 256,
                                      output_size = 15,
                                      weight_init = 0.01 
                                      )
                        
trainer = trainer.Trainer(network, 
                          train_images, train_labels, test_images, test_labels,
                          epochs = max_epochs, 
                          mini_batch_size = 10,
                          optimizer = 'Adam', 
                          optimizer_param = {'lr': 0.5},
                          evaluate_sample_num_per_epoch = 1000
                          )
trainer.train()


network.save_params("params.pkl")
print("Saved Network Parameters!")