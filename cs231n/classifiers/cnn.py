from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        w1 = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        b1 = np.zeros((1, num_filters))
        
        pad = (filter_size - 1) // 2
        stride = 1
        outw = (W + 2 * pad - filter_size) // stride + 1
        #print(('W--: ', W))
        #print(('filter_size--: ', filter_size))
        #print(('outw--: ', outw))
        h1InputNum = outw * outw * num_filters // 4
        w2 = weight_scale * np.random.randn(h1InputNum, hidden_dim)
        b2 = np.zeros((1, hidden_dim))
        
        w3 = weight_scale * np.random.randn(hidden_dim, num_classes)
        #print(('w3--: ', w3.shape))
        b3 = np.zeros((1, num_classes))
        self.params = {'W1' : w1, 'b1' : b1, 'W2' : w2, 'b2' : b2, 'W3' : w3, 'b3' : b3}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #out_convtest, cachetest = conv_forward_naive(X, W1, b1, conv_param)
        #print(('out_convtest.shape: ', out_convtest.shape))
        out_conv, cache = conv_relu_forward(X, W1, b1, conv_param)
        out_max, cache_max = max_pool_forward_naive(out_conv, pool_param)
        #out_max = np.reshape(out_max, (out_max.shape[0], -1))

        out2, cache2 = affine_relu_forward(out_max, self.params['W2'], self.params['b2'])
        scores, cache3 = affine_forward(out2, self.params['W3'], self.params['b3'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        dims = X.shape[1:]
        xr = np.reshape(X, (X.shape[0], np.prod(dims)))
        
        dataloss, dataDx = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2']) + 0.5 * self.reg * np.sum(self.params['W3'] * self.params['W3'])
        loss= dataloss + reg_loss
        #print(('loss is: ', loss))
        
        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW3 = np.dot(out2.T, dataDx)
        #print(('dW3----: ', dW3.shape))
        db3 = np.sum(dataDx, axis=0, keepdims=True)
        dout2 = np.dot(dataDx, self.params['W3'].T)
        dout2[out2 <= 0] = 0
        #print(('dout2: ', dout2.shape))
        
        # next backprop into hidden layer
        
        # backprop the ReLU non-linearity
        
        # finally into W1,b1
        dW2 = np.dot(np.reshape(out_max, (out_max.shape[0], -1)).T, dout2)
        #print(('dout2--: ', dout2.shape))
        #print(('out_max.T--: ', out_max.T.shape))
        db2 = np.sum(dout2, axis=0, keepdims=True)
        dmax = np.dot(dout2, self.params['W2'].T)
        
        #print(('dmax--: ', dmax.shape))
        dmax = np.reshape(dmax, out_max.shape)
        #dx, dW1, db1 = conv_relu_pool_backward(dmax, cache_max)
        dconv = max_pool_backward_naive(dmax, cache_max)
        
        dx, dW1, db1 = conv_relu_backward(dconv, cache)
  
        # add regularization gradient contribution
        dW3 += self.reg * self.params['W3']
        #print(('dW3: ', dW3.shape))
        #print(('self.params_W3: ', self.params['W3'].shape))
        #print(('dW1: ', dW1.shape))
        #print(('self.params_W1: ', self.params['W1'].shape))
        dW1 = dW1 + self.reg * self.params['W1']
        dW2 = np.reshape(dW2, W2.shape) + self.reg * self.params['W2']
        grads = {'W1':dW1, 'b1':db1, 'W2':dW2, 'b2':db2, 'W3':dW3, 'b3':db3}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
