# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.state = x      # for backprop use
        output_size = (x.shape[2] - self.kernel_size) // self.stride + 1
        batch_size = x.shape[0]
        out = np.zeros((batch_size, self.out_channel, output_size))
        for i in range(output_size):
            out[:,:,i] = np.tensordot(x[:,:,i*self.stride:i*self.stride+self.kernel_size], self.W, axes=([1,2],[1,2]))
        return out + self.b.reshape((1, self.out_channel, 1))

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size = delta.shape[0]
        output_size = delta.shape[2]
        input_size = self.state.shape[2]
        # perform dilation on delta when stride > 1
        if(self.stride > 1):
            delta_temp = np.zeros((batch_size, self.out_channel, (output_size-1)*self.stride+1))
            for i in range(output_size):
                delta_temp[:,:,i*self.stride] = delta[:,:,i]
            delta = delta_temp
        # when stride > 1, some input may not be involved
        # may need to pad some more 0
        extra = input_size - (self.stride * (output_size-1) + self.kernel_size)
        padded_delta = np.pad(delta, pad_width=((0,0), (0,0), (self.kernel_size-1, self.kernel_size-1+extra)), mode="constant")
        flipped_kernel = self.W[:,:,::-1]
        dx = np.zeros(self.state.shape)
        for i in range(input_size):
            dx[:,:,i] = np.tensordot(padded_delta[:,:,i:i+self.kernel_size], flipped_kernel, axes=([1,2],[0,2]))
        
        # calculate unaveraged gradient for parameters
        for i in range(self.kernel_size):
            self.dW[:,:,i] = np.tensordot(delta, self.state[:,:,i:i+delta.shape[2]], axes=([0,2],[0,2]))
        self.db = np.sum(np.sum(delta, axis=2), axis=0)

        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        # use same stride and kernel size in both directions
        self.kernel_size = kernel_size
        self.stride = stride

        # set default to Kaiming init
        if weight_init_fn is None:
            bound = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
            self.W = np.random.uniform(-1 * bound, bound, size=(out_channel, in_channel, kernel_size, kernel_size))
            # self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.random.uniform(-1 * bound, bound, size=out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.state = x      # for backprop use
        output_width = (x.shape[2] - self.kernel_size) // self.stride + 1
        output_height = (x.shape[3] - self.kernel_size) // self.stride + 1
        batch_size = x.shape[0]
        out = np.zeros((batch_size, self.out_channel, output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                out[:,:,i,j] = np.tensordot(x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size], self.W, axes=([1,2,3],[1,2,3]))
        return out + self.b.reshape((1, self.out_channel, 1, 1))
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size = delta.shape[0]
        output_width = delta.shape[2]
        output_height = delta.shape[3]
        input_width = self.state.shape[2]
        input_height = self.state.shape[3]
        # perform dilation on delta when stride > 1
        if(self.stride > 1):
            delta_temp = np.zeros((batch_size, self.out_channel, (output_width-1)*self.stride+1, (output_height-1)*self.stride+1))
            for i in range(output_width):
                for j in range(output_height):
                    delta_temp[:,:,i*self.stride,j*self.stride] = delta[:,:,i,j]
            delta = delta_temp
        # when stride > 1, some input may not be involved
        # may need to pad some more 0
        extra_width = input_width - (self.stride * (output_width-1) + self.kernel_size)
        extra_height = input_height - (self.stride * (output_height-1) + self.kernel_size)
        padded_delta = np.pad(delta, pad_width=((0,0), (0,0), (self.kernel_size-1, self.kernel_size-1+extra_width), (self.kernel_size-1, self.kernel_size-1+extra_height)), mode="constant")
        flipped_kernel = self.W[:,:,::-1,::-1]
        dx = np.zeros(self.state.shape)
        for i in range(input_width):
            for j in range(input_height):
                dx[:,:,i,j] = np.tensordot(padded_delta[:,:,i:i+self.kernel_size,j:j+self.kernel_size], flipped_kernel, axes=([1,2,3],[0,2,3]))
        
        # calculate unaveraged gradient for parameters
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dW[:,:,i,j] = np.tensordot(delta, self.state[:,:,i:i+delta.shape[2],j:j+delta.shape[3]], axes=([0,2,3],[0,2,3]))
        self.db = np.sum(np.sum(np.sum(delta, axis=3), axis=2), axis=0)

        return dx
        

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c*self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)
