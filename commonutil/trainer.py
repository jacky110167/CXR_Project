import sys, os
sys.path.append(os.pardir)
import numpy as np
from commonutil.optimizer import *

class Trainer:
    def __init__(self, network, 
                 train_images, train_labels, test_images, test_labels,
                 epochs = 20, 
                 mini_batch_size = 20,
                 optimizer='Adam', 
                 optimizer_param = {'lr':0.01}, 
                 evaluate_sample_num_per_epoch = None, 
                 verbose = True
                 ):
        
        self.network = network
        self.verbose = verbose
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images  = test_images
        self.test_labels  = test_labels
        
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        
        optimizer_class_dict = {'sgd':SGD, 
                                'momentum':Momentum, 
                                'nesterov':Nesterov,
                                'adagrad':AdaGrad, 
                                'rmsprpo':RMSprop, 
                                'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = train_images.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        print('Current iterations: ', self.current_iter +1)
        
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.train_images[batch_mask]
        t_batch = self.train_labels[batch_mask]
        
        grads = self.network.backprop(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: 
            print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.train_images, self.train_labels
            x_test_sample , t_test_sample  = self.test_images , self.test_labels
            
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.train_images[:t], self.train_labels[:t]
                x_test_sample , t_test_sample  = self.test_images[:t] , self.test_labels[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc  = self.network.accuracy(x_test_sample, t_test_sample)
            
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            
            
            if self.verbose:
                print("=== epoch: " + str(self.current_epoch) + ", train acc: " + str(train_acc) + ", test acc: " + str(test_acc) + " ===")
        
        self.current_iter += 1

    def train(self):
        for i in range( self.max_iter ):
            self.train_step()

        test_acc = self.network.accuracy( self.test_images , self.test_labels)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))