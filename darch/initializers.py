import numpy as np
import tensorflow as tf

def zeros_initializer():
    def init_fn(shape):
        init_vals = tf.zeros_initializer(shape)
        return init_vals
    return init_fn

def constant_initializer(val):
    initer = tf.constant_initializer(val)
    def init_fn(shape):
        init_vals = initer(shape)
        return init_vals
    return init_fn

def gaussian_initializer(mean=0.0, stddev=1.0):
    initer = tf.random_normal_initializer(mean, stddev)
    def init_fn(shape):
        init_vals = initer(shape)
        return init_vals
    return init_fn

def xavier_initializer_affine(gain=1.0):
    def init_fn(shape):
        n, m = shape
        
        sc = gain * ( np.sqrt(6.0) / np.sqrt(m + n) )
        init_vals = tf.random_uniform([n, m], -sc, sc)
        return init_vals
    return init_fn

def kaiming2015delving_initializer_conv(gain=1.0):
    """Initializer proposed in the paper "Delving Deep into Rectifiers" 
    for initializing conv layers with RELU nonlinearities. 

    """

    def init_fn(shape):
        n = np.product(shape)
        stddev = gain * np.sqrt( 2.0 / n )
        init_vals = tf.random_normal(shape, 0.0, stddev)
        return init_vals
    return init_fn

# similar to a kaiming's initialization; only more descriptive name.
def invsqrt_size_gaussian_initializer_conv(gain=1.0):
    def init_fn(shape):
        n = np.product(shape)
        stddev = gain * np.sqrt( 1.0 / n )
        init_vals = tf.random_normal(shape, 0.0, stddev)
        return init_vals
    return init_fn

def invsqrt_size_gaussian_initializer_affine(gain=1.0):
    def init_fn(shape):
        n, _  = shape
        stddev = gain * np.sqrt( 1.0 / n )
        init_vals = tf.random_normal(shape, 0.0, stddev)
        return init_vals
    return init_fn

