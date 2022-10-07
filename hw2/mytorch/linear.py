# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)
        # the testcase of hw2 section6 incorrectly initialize self.b to shape (out feature,)
        if(len(self.b.shape) == 1):
            self.b = np.expand_dims(self.b, axis=0)
        self.x = None

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x
        return self.x.dot(self.W) + self.b

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        # here x and y are arranged as row vector
        # and the shape of W is in*out, not out*in as in lecture
        # so from the perspective of formula they are all transposed
        # the forward is thus yT=xTWT+bT for each x
        # so if written as formula, dW here would be dWT in the slides, the average of all dL/dyT * dyT/dWT(which is xT) in the batch
        # dL/dyT is one row of delta transposed
        # the same for other derivatives
        # the shape of dW should be that of WT, here it's transposed to be the same as the shape of W
        print(delta.shape)
        print(self.db.shape)
        batch_size = delta.shape[0]
        for i in range(batch_size):
            self.dW += np.dot(np.expand_dims(delta[i], axis=0).T, np.expand_dims(self.x[i], axis=0)).T
            self.db += np.expand_dims(delta[i], axis=0)
        self.dW /= batch_size
        self.db /= batch_size

        return np.dot(delta, self.W.T)
