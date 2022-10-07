# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        self.x = x

        if eval:
            self.mean = self.running_mean
            self.var = self.running_var
        else:
            self.mean[0] = np.average(self.x, axis=0)
            self.var[0] = np.average((self.x - self.mean)**2, axis=0)
        self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.norm * self.gamma + self.beta

        # Update running batch statistics
        if not eval:
            self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        batch_size = delta.shape[0]
        self.dbeta[0] = np.sum(delta, axis=0)
        self.dgamma[0] = np.sum(delta*self.norm, axis=0)
        dnorm = delta * self.gamma
        dvar = -0.5 * np.sum(dnorm * (self.x - self.mean) * (self.var + self.eps)**(-1.5), axis=0)
        dvar = np.expand_dims(dvar, axis=0)     # convert shape from (in feature, ) to (1, in feature)
        dmean = -1 * np.sum(dnorm * (self.var + self.eps)**(-0.5), axis=0) - 2/batch_size * dvar[0] * np.sum(self.x - self.mean, axis=0)
        dmean = np.expand_dims(dmean, axis=0)
        dx = dnorm * (self.var + self.eps)**(-0.5) + dvar * 2/batch_size * (self.x - self.mean) + dmean / batch_size
        return dx
