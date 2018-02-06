import numpy as np
from commonutil import util
from commonutil import functions

class Convolutional_Layer:
    def __init__(self, w, b, stride=1):
        self.w = w
        self.b = b
        self.stride = stride
        
        self.x = None   
        self.col_x = None
        self.col_w = None

        self.dw = None
        self.db = None

    def forward(self, x):
        fn, fc, fh, fw = self.w.shape
        n, c, h, w = x.shape
        
        out_h = util.conv_output_size(h, fh, self.stride)
        out_w = util.conv_output_size(w, fw, self.stride)
        
        col_x = util.im2col(x, fh, fw, self.stride)
        col_w = self.w.reshape(fn, -1).T
        
        output = np.dot(col_x, col_w) + self.b
        output = output.reshape(n, out_h, out_w, fn).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col_x = col_x
        self.col_w = col_w

        return output

    def backward(self, dout):
        fn, fc, fh, fw = self.w.shape
        
        dout = dout.transpose(0 ,2 ,3 ,1).reshape(-1, fn)

        self.db = np.sum(dout, axis=0)

        self.dw = np.dot(self.col_x.T, dout)
        self.dw = self.dw.reshape(fn, fc, fh, fw)

        dcol_x = np.dot(dout, self.col_w.T)
        dx = util.col2im(dcol_x, self.x, fh, fw, self.stride)
        
        return dx

    
class Relu_Layer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
class Pooling_Layer:
    def __init__(self, ph, pw, stride = 1):
        self.ph = ph
        self.pw = pw
        self.stride = stride
        
        self.x = None
        self.arg_max = None
        
    def forward(self, x):
        n, c, h, w = x.shape
        
        out_h = util.conv_output_size(h, self.ph, self.stride)
        out_w = util.conv_output_size(w, self.pw, self.stride)
        
        col_x = util.im2col(x, self.ph, self.pw, self.stride)
        col_x = col_x.reshape(-1, self.ph * self.pw)
        
        arg_max = np.argmax(col_x, axis = 1)
        output = np.max(col_x, axis = 1)
        output = output.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)
        
        self.x = x
        self.arg_max = arg_max
        
        return output
    
    def backward(self, dout):
        pool_size = self.ph * self.pw
        
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = util.col2im(dcol, self.x, self.ph, self.pw, self.stride)
        
        return dx
    
class Affine_Layer:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)
        
        return dx

class Sigmoid_Layer_with_CEE:
    def __init__(self):
        self.y = None
        self.t = None

        self.loss = None
        
    def forward(self, x, t):
        self.y = functions.sigmoid(x)
        self.t = t   
        self.loss = functions.cross_entropy_error(self.y, self.t)
        
        return self.loss
        
    '''Need to checked'''
    def backward(self, dout = 1):
        batchSize = self.t.shape[0]
        dx = (self.y - self.t) / batchSize

        return dx

class Dropout_Layer:

    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
