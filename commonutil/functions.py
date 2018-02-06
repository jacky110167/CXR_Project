import numpy as np


def cross_entropy_error(y , t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
             
    batchSize = y.shape[0]
    
    return  np.sum( -t*np.log( y + 1e-7) - (1-t)*np.log( 1-y + 1e-7) ) / batchSize


def sigmoid(x): 
    return 1 / (1 + np.exp(-x))