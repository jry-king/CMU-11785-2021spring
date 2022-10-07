import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        # their shapes are all (hidden_dim)
        self.r = self.r_act(np.dot(self.Wrx, x) + self.bir + np.dot(self.Wrh, h) + self.bhr)
        self.z = self.z_act(np.dot(self.Wzx, x) + self.biz + np.dot(self.Wzh, h) + self.bhz)
        self.n = self.h_act(np.dot(self.Wnx, x) + self.bin + self.r * (np.dot(self.Wnh, h) + self.bhn))
        h_t = (1 - self.z) * self.n + self.z * h

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs match the
        # initalized shapes accordingly

        # starting from deriving derivatives from h_t
        # actually takes delta of shape (1, hidden) as input
        if delta.shape[0] == 1:
            delta = np.squeeze(delta, axis=0)
        dn = delta * (1 - self.z)                   # (hidden)
        dz = delta * (-1 * self.n + self.hidden)    # (hidden)
        dh = delta * self.z                         # (hidden)
        dh = np.expand_dims(dh, axis=0)             # (1, hidden)
        # derive from n
        dnz = dn * self.h_act.derivative()                                                  # (hidden)
        dx = np.dot(np.expand_dims(dnz, axis=0), self.Wnx)                                  # (1, input)
        self.dWnx = np.dot(np.expand_dims(self.x, axis=1), np.expand_dims(dnz, axis=0)).T   # (hidden, input)
        self.dbin = dnz                                                                     # (hidden)
        dr = dnz * (np.dot(self.Wnh, self.hidden) + self.bhn)                                               # (hidden)
        dh += np.dot(np.expand_dims(dnz * self.r, axis=0), self.Wnh)                                        # (1, hidden)
        self.dWnh = np.dot(np.expand_dims(self.hidden, axis=1), np.expand_dims(dnz * self.r, axis=0)).T     # (hidden, input)
        self.dbhn = dnz * self.r                                                                            # (hidden)
        # derive from z
        dzz = dz * self.z_act.derivative()
        dx += np.dot(np.expand_dims(dzz, axis=0), self.Wzx)
        self.dWzx = np.dot(np.expand_dims(self.x, axis=1), np.expand_dims(dzz, axis=0)).T
        self.dbiz = dzz
        dh += np.dot(np.expand_dims(dzz, axis=0), self.Wzh)
        self.dWzh = np.dot(np.expand_dims(self.hidden, axis=1), np.expand_dims(dzz, axis=0)).T
        self.dbhz = dzz
        # derive from r
        drz = dr * self.r_act.derivative()
        dx += np.dot(np.expand_dims(drz, axis=0), self.Wrx)
        self.dWrx = np.dot(np.expand_dims(self.x, axis=1), np.expand_dims(drz, axis=0)).T
        self.dbir = drz
        dh += np.dot(np.expand_dims(drz, axis=0), self.Wrh)
        self.dWrh = np.dot(np.expand_dims(self.hidden, axis=1), np.expand_dims(drz, axis=0)).T
        self.dbhr = drz

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
