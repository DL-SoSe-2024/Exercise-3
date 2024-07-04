import numpy as np
from Layers import Base
import scipy.signal
import copy

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, tuple([self.num_kernels] + list(self.convolution_shape)))
        self.bias = np.random.uniform(0, 1, self.num_kernels)
        self.input_tensor = None
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output = []
        for k in range(self.num_kernels):
            kernel = self.weights[k,...]
            expanded_kernel = np.expand_dims(kernel, axis=0)
            padding = ((0,0),(0,0),) + tuple((int(np.ceil((kernel.shape[i]-1)/2)), int(np.floor((kernel.shape[i]-1)/2))) for i in range(1, len(kernel.shape)))
            padded_array = np.pad(input_tensor, padding)
            output_layer = scipy.signal.correlate(padded_array, expanded_kernel, mode='valid', method='auto') + self.bias[k]
            out_shape = (output_layer.shape[0], output_layer.shape[1]) + tuple((1 + (input_tensor.shape[i]+ int(np.ceil((expanded_kernel.shape[i]-1)/2)) + int(np.floor((expanded_kernel.shape[i]-1)/2)) - expanded_kernel.shape[i])//self.stride_shape[i-2] for i in range(2, len(output_layer.shape))))
            out_strides = (output_layer.strides[0], output_layer.strides[1]) + tuple((output_layer.strides[i] * self.stride_shape[i-2] for i in range(2, len(output_layer.shape))))
            output.append(np.lib.stride_tricks.as_strided(output_layer, shape=out_shape, strides=out_strides))
        output = np.concatenate(output, axis=1)
        return output
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.bias_optimizer = copy.deepcopy(optimizer)
        
    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    
    @bias_optimizer.setter
    def bias_optimizer(self, bias_optimizer):
        self._bias_optimizer = bias_optimizer
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
        
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias    
    
    def backward(self, error_tensor):
        #Upsampling
        x = [i for i in range(1,error_tensor.shape[2])]
        error_tensor = np.insert(error_tensor,[copy.copy(e) for _ in range(self.stride_shape[0]-1) for e in x],0,axis=2)
        if len(error_tensor.shape) > 3:
            y = [j for j in range(1,error_tensor.shape[3])]
            error_tensor = np.insert(error_tensor,[copy.copy(e) for _ in range(self.stride_shape[1]-1) for e in y],0,axis=3)
        padding = ((0,0),(0,0),) + tuple((0,self.input_tensor.shape[i] - error_tensor.shape[i]) for i in range(2,len(error_tensor.shape)))
        error_tensor = np.pad(error_tensor, padding)
        

        #Transpose and Flip Weights
        if len(self.weights.shape) < 4:
            weights_t = np.transpose(self.weights,(1,0,2))
        else:
            weights_t = np.transpose(self.weights,(1,0,2,3))
        weights_t = np.flip(weights_t, axis=1)    

        # E_n = W^T * E_n-1
        err_prev = []
        for i in range(error_tensor.shape[0]):
            batch = error_tensor[i,...]
            err_out_channel = []
            for k in range(weights_t.shape[0]):
                kernel = weights_t[k,...]
                padded_batch = np.pad(batch, ((0,0),) + tuple((int(np.ceil((kernel.shape[i]-1)/2)), int(np.floor((kernel.shape[i]-1)/2))) for i in range(1, len(kernel.shape))))
                err_out_layer = scipy.signal.convolve(padded_batch, kernel, mode='valid', method='auto')
                err_out_channel.append(err_out_layer)
            err_prev.append(np.concatenate(err_out_channel, axis=0))
        err_prev = np.stack(err_prev, axis = 0)
        
        # DW = En * X^T
        padded_input = np.pad(self.input_tensor,((0,0), (0,0),) + tuple((int(np.ceil((self.weights.shape[i]-1)/2)), int(np.floor((self.weights.shape[i]-1)/2))) for i in range(2, len(self.weights.shape))))
        gradient_kernel = []
        for i in range(error_tensor.shape[1]):
            gradient_kernel.append(scipy.signal.correlate(padded_input, error_tensor[:,[i],...], mode='valid', method='auto'))
        self.gradient_weights = np.concatenate(gradient_kernel, axis=0)
        
        #bias
        gradient_bias_array = []
        for i in range(error_tensor.shape[1]):
            err_next_channel = error_tensor[:,i,...]
            gradient_bias_array.append(err_next_channel.sum())
        self.gradient_bias = np.array(gradient_bias_array)
        
        #calculate updates for weight and bias
        if hasattr(self, "optimizer"):
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return err_prev
    
    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        weights_shape = tuple([self.num_kernels,] + list(self.convolution_shape))
        self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        bias_shape = self.num_kernels
        self.bias = bias_initializer.initialize(bias_shape,fan_in,fan_out)