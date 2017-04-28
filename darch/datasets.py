
import numpy as np
import scipy as sp
import tensorflow as tf
import cPickle
import os

class InMemoryDataset:
    """Wrapper around a dataset for iteration that allows cycling over the 
    dataset. 

    This functionality is especially useful for training. One can specify if 
    the data is to be shuffled at the end of each epoch. It is also possible
    to specify a transformation function to applied to the batch before
    being returned by next_batch.

    """
    
    def __init__(self, X, y, shuffle_at_epoch_begin, batch_transform_fn=None):
        if X.shape[0] != y.shape[0]:
            assert ValueError("X and y the same number of examples.")

        self.X = X
        self.y = y
        self.shuffle_at_epoch_begin = shuffle_at_epoch_begin
        self.batch_transform_fn = batch_transform_fn
        self.iter_i = 0

    def get_num_examples(self):
        return self.X.shape[0]

    def next_batch(self, batch_size):
        """Returns the next batch in the dataset. 

        If there are fewer that batch_size examples until the end
        of the epoch, next_batch returns only as many examples as there are 
        remaining in the epoch.

        """

        n = self.X.shape[0]
        i = self.iter_i

        # shuffling step.
        if i == 0 and self.shuffle_at_epoch_begin:
            inds = np.random.permutation(n)
            self.X = self.X[inds]
            self.y = self.y[inds]

        # getting the batch.
        eff_batch_size = min(batch_size, n - i)
        X_batch = self.X[i:i + eff_batch_size]
        y_batch = self.y[i:i + eff_batch_size]
        self.iter_i = (self.iter_i + eff_batch_size) % n

        # transform if a transform function was defined.
        if self.batch_transform_fn != None:
            X_batch_out, y_batch_out = self.batch_transform_fn(X_batch, y_batch)
        else:
            X_batch_out, y_batch_out = X_batch, y_batch

        return (X_batch_out, y_batch_out)

def load_mnist(data_dir, flatten=False, one_hot=True, normalize_range=False):
    from tensorflow.examples.tutorials.mnist import input_data
    # print data_dir
    mnist = input_data.read_data_sets(data_dir, one_hot=one_hot, reshape=flatten)

    def _extract_fn(x):
        X = x.images
        y = x.labels

        if not normalize_range: 
            X *= 255.0
        
        return (X, y)
        
    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def load_cifar10(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=True, border_pad_size=0):
    """Loads all of CIFAR-10 in a numpy array.

    Provides a few options for the output formats. For example, 
    normalize_range returns the output images with pixel values in [0.0, 1.0].
    The other options are self explanatory. Border padding corresponds to 
    upsampling the image by zero padding the border of the image.

    """
    train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
    val_filenames = ['data_batch_5']
    test_filenames = ['test_batch']

    # NOTE: this function uses some arguments from the outer scope, namely 
    # flatten, one_hot, normalize_range, and possibly others once added.
    def _load_data(fpath):
        with open(fpath, 'rb') as f: 
            d = cPickle.load(f)

            # for the data
            X = d['data'].astype('float32')

            # reshape the data to the format (num_images, height, width, depth) 
            num_images = X.shape[0]
            num_classes = 10
            X = X.reshape( (num_images, 3, 32, 32) )
            X = X.transpose( (0,2,3,1) )
            X = X.astype('float32')
            
            # transformations based on the argument options.
            if normalize_range:
                X = X / 255.0
            
            if flatten:
                X = X.reshape( (num_images, -1) )

            # for the labels
            y = np.array(d['labels'])

            if one_hot:
                y_one_hot = np.zeros( (num_images, num_classes), dtype='float32')
                y_one_hot[ np.arange(num_images),  y ] = 1.0
                y = y_one_hot

            return (X, y)

    # NOTE: this function uses some arguments from the outer scope.
    def _load_data_multiple_files(fname_list):

        X_parts = []
        y_parts = []
        for fname in fname_list:
            fpath = os.path.join(data_dir, fname)
            X, y = _load_data(fpath)
            X_parts.append(X)
            y_parts.append(y)

        X_full = np.concatenate(X_parts, axis=0)
        y_full = np.concatenate(y_parts, axis=0)

        return (X_full, y_full)
    
    Xtrain, ytrain = _load_data_multiple_files(train_filenames)
    Xval, yval = _load_data_multiple_files(val_filenames)
    Xtest, ytest = _load_data_multiple_files(test_filenames)

    if whiten_pixels:
        mean = Xtrain.mean(axis=0)[None, :]
        std = Xtrain.std(axis=0)[None, :]
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = zero_pad_border(Xtrain, border_pad_size)
        Xval = zero_pad_border(Xval, border_pad_size)
        Xtest = zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def onehot_to_idx(y_onehot):
    y_idx = np.where(y_onehot > 0.0)[1]

    return y_idx

def idx_to_onehot(y_idx, num_classes):
    num_images = y.shape[0]
    y_one_hot = np.zeros( (num_images, num_classes), dtype='float32')
    y_one_hot[ np.arange(num_images),  y ] = 1.0

    return y_one_hot

def center_crop(X, out_height, out_width):
    num_examples, in_height, in_width, in_depth = X.shape
    assert out_height <= in_height and out_width <= in_width

    start_i = (in_height - out_height) / 2
    start_j = (in_width - out_width) / 2
    out_X = X[:, start_i : start_i + out_height, start_j : start_j + out_width, :]  

    return out_X

# random crops for each of the images.
def random_crop(X, out_height, out_width):
    num_examples, in_height, in_width, in_depth = X.shape
    # the ouput dimensions have to be smaller or equal that the input dimensions.
    assert out_height <= in_height and out_width <= in_width

    start_is = np.random.randint(in_height - out_height + 1, size=num_examples)
    start_js = np.random.randint(in_width - out_width + 1, size=num_examples)
    out_X = []
    for ind in xrange(num_examples):
        st_i = start_is[ind]
        st_j = start_js[ind]

        out_Xi = X[ind, st_i : st_i + out_height, st_j : st_j + out_width, :]
        out_X.append(out_Xi)

    out_X = np.array(out_X)
    return out_X

def random_flip_left_right(X, p_flip):
    num_examples, height, width, depth = X.shape

    out_X = X.copy()
    flip_mask = np.random.random(num_examples) < p_flip
    out_X[flip_mask] = out_X[flip_mask, :, ::-1, :]

    return out_X

def per_image_whiten(X):
    """ Subtracts the mean of each image in X and renormalizes them to unit norm.

    """
    num_examples, height, width, depth = X.shape

    X_flat = X.reshape((num_examples, -1))
    X_mean = X_flat.mean(axis=1)
    X_cent = X_flat - X_mean[:, None]
    X_norm = np.sqrt( np.sum( X_cent * X_cent, axis=1) ) 
    X_out = X_cent / X_norm[:, None]
    X_out = X_out.reshape(X.shape) 

    return X_out

# Assumes the following ordering for X: (num_images, height, width, num_channels)
def zero_pad_border(X, pad_size):
    n, height, width, num_channels = X.shape
    X_padded = np.zeros((n, height + 2 * pad_size, width + 2 * pad_size, 
        num_channels), dtype='float32')
    X_padded[:, pad_size:height + pad_size, pad_size:width + pad_size, :] = X
    
    return X_padded

# auxiliary functions for 
def get_augment_cifar_data_train(out_height, out_width, p_flip):
    def augment_fn(X, y):
        X_out = random_crop(X, out_height, out_width)
        X_out = random_flip_left_right(X_out, p_flip)
        #X_out = per_image_whiten(X_out)
        y_out = y

        return (X_out, y_out)
    return augment_fn

# for evaluation, crop in the middle and do per image "whitening".
def get_augment_cifar_data_eval(out_height, out_width):
    def augment_fn(X, y):
        X_out = center_crop(X, out_height, out_width)
        #X_out = per_image_whiten(X_out)
        y_out = y

        return (X_out, y_out)
    return augment_fn


