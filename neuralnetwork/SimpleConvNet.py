import numpy as np
from collections import OrderedDict
from commonutil import layers
import pickle

class SimpleConvNet:
    def __init__(self, 
                 input_dim = (1, 1024, 1024),
                 conv_params = {
                                'filter_num' : (64, 128, 128, 256),
                                'filter_size': (13,   7,   5,   3),
                                'stride'     : ( 3,   2,   1,   1),
                                },
                 hidden_size = 256,
                 output_size = 15,
                 weight_init = 0.01 
                 ):
        
        fn1 = conv_params['filter_num'][0]
        fs1 = conv_params['filter_size'][0]
        stride1 = conv_params['stride'][0]
        
        fn2 = conv_params['filter_num'][1]
        fs2 = conv_params['filter_size'][1]
        stride2 = conv_params['stride'][1]
        
        fn3 = conv_params['filter_num'][2]
        fs3 = conv_params['filter_size'][2]
        stride3 = conv_params['stride'][2]
        
        fn4 = conv_params['filter_num'][3]
        fs4 = conv_params['filter_size'][3]
        stride4 = conv_params['stride'][3]
      
        pool_output_size = 256*8*8
        
        # Parameters
        self.params = {} 
        
        self.params['w1'] = weight_init * np.random.randn(fn1, input_dim[0], fs1, fs1)
        self.params['b1'] = np.zeros(fn1)
        
        self.params['w2'] = weight_init * np.random.randn(fn2, fn1, fs2, fs2)
        self.params['b2'] = np.zeros(fn2)
        
        self.params['w3'] = weight_init * np.random.randn(fn3, fn2, fs3, fs3)
        self.params['b3'] = np.zeros(fn3)
        
        self.params['w4'] = weight_init * np.random.randn(fn4, fn3, fs4, fs4)
        self.params['b4'] = np.zeros(fn4)
        
        self.params['w5'] = weight_init * np.random.randn(pool_output_size, hidden_size)
        self.params['b5'] = np.zeros(hidden_size)
        
        self.params['w6'] = weight_init * np.random.randn(hidden_size, output_size)
        self.params['b6'] = np.zeros(output_size)
        
        
        # Network structure
        self.layers = OrderedDict()
        
        #self.layers['Input']
        
        self.layers['Conv1']   = layers.Convolutional_Layer(self.params['w1'],
                                                            self.params['b1'],
                                                            stride1)
        self.layers['Relu1']   = layers.Relu_Layer()
        self.layers['Pool1']   = layers.Pooling_Layer(ph = 2, pw = 2, stride = 2)
        
        
        self.layers['Conv2']   = layers.Convolutional_Layer(self.params['w2'],
                                                            self.params['b2'],
                                                            stride2)
        self.layers['Relu2']   = layers.Relu_Layer()
        self.layers['Pool2']   = layers.Pooling_Layer(ph = 2, pw = 2, stride = 2)
        
        
        self.layers['Conv3']   = layers.Convolutional_Layer(self.params['w3'],
                                                            self.params['b3'],
                                                            stride3)
        self.layers['Relu3']   = layers.Relu_Layer()
        self.layers['Pool3']   = layers.Pooling_Layer(ph = 2, pw = 2, stride = 2)
        
        
        self.layers['Conv4']   = layers.Convolutional_Layer(self.params['w4'],
                                                            self.params['b4'],
                                                            stride4)
        self.layers['Relu4']   = layers.Relu_Layer()
        self.layers['Pool4']   = layers.Pooling_Layer(ph = 2, pw = 2, stride = 2)
        
        
        self.layers['Affine1'] = layers.Affine_Layer(self.params['w5'], self.params['b5'])
        self.layers['Relu5']   = layers.Relu_Layer()
        #self.layers['Dropout1']= layers.Dropout_Layer(0.6)
        
        
        self.layers['Affine2'] = layers.Affine_Layer(self.params['w6'], self.params['b6'])
        self.layers['Relu6']   = layers.Relu_Layer()
        
        self.last_layer  = layers.Sigmoid_Layer_with_CEE()
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    
    def loss(self, x ,t):
        y = self.predict(x)
        
        return self.last_layer.forward(y, t)
    
    
    def backprop(self, x, t):
        #forward propagation
        self.loss(x, t)
        
        #backward propagation
        dout = 1
        dout = self.last_layer.backward(dout)
        
        Rlayers = list(self.layers.values())
        Rlayers.reverse()
        
        for Rlayer in Rlayers:
            dout = Rlayer.backward(dout)
            
        
        gradients = {}
        gradients['w1'], gradients['b1'] = self.layers['Conv1'].dw  , self.layers['Conv1'].db
        gradients['w2'], gradients['b2'] = self.layers['Conv2'].dw  , self.layers['Conv2'].db
        gradients['w3'], gradients['b3'] = self.layers['Conv3'].dw  , self.layers['Conv3'].db
        gradients['w4'], gradients['b4'] = self.layers['Conv4'].dw  , self.layers['Conv4'].db
        gradients['w5'], gradients['b5'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
        gradients['w6'], gradients['b6'] = self.layers['Affine2'].dw, self.layers['Affine2'].db
        
        return gradients
    
    
    def accuracy(self, x, t, batchSize = 100):
        
        accuracy = 0.0
        
        for i in range(int(x.shape[0] / batchSize)):
            tx = x[i*batchSize : (i+1)*batchSize]
            tt = t[i*batchSize : (i+1)*batchSize]
            
            y = self.predict(tx)
            y_transform = (y > 0.5).astype(np.int)
            
            for k in range(batchSize):
                accuracy += np.array_equal(y_transform[k], tt[k])
        
        return accuracy / x.shape[0]
    
    def save_params(self, file_name = 'params.pkl'):
        params = {}
        
        for key, val in self.params.items():
            params[key] = val
        
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    

    def load_params(self, file_name = 'params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        for key, val in params.items():
            self.params[key] = val
            
        for idx, key in enumerate(['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Affine1', 'Affine2']):
            self.layers[key].w = self.params['w' + str(idx + 1)]
            self.layers[key].b = self.params['b' + str(idx + 1)]