import numpy as np
from Layers import Base, Helpers

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.weights = None
        self.bias = None
        self.moving_mean = None
        self.mean = None
        self.var = None
        self.moving_var = None
        self.alpha = 0.8
        self.input_tensor_normalized = None
        self.input_tensor = None
        self._optimizer = None
        self.gradient_bias = None
        self.gradient_weights = None
        self.original_input_shape = None
    
    def forward(self, input_tensor):
        output_tensor = None
        reformatted = False
        # for reformating later
        self.original_input_shape = input_tensor.shape
        

        if input_tensor.ndim == 4:
            input_tensor = self.reformat(input_tensor)
            reformatted = True

        if input_tensor.ndim == 2:
            self.input_tensor = input_tensor
            self.mean = np.mean(input_tensor, axis = 0, dtype=float, keepdims=True)
            self.var = np.var(input_tensor, axis = 0, dtype=float, keepdims=True)

            if self.weights is None or self.bias is None:
                self.initialize()

            if not self.testing_phase:
                if self.moving_mean is None:
                    self.moving_mean = self.mean
                    self.moving_var = self.var
                else:
                    self.moving_mean = self.alpha * self.moving_mean + (1-self.alpha) * self.mean
                    self.moving_var = self.alpha * self.moving_var + (1-self.alpha) * self.var
                self.input_tensor_normalized = np.divide((input_tensor - self.mean), np.sqrt(self.var + np.finfo(float).eps)) 
                output_tensor = np.multiply(self.weights, self.input_tensor_normalized) + self.bias  
            else:
                self.input_tensor_normalized = np.divide((input_tensor - self.moving_mean), 
                np.sqrt(self.moving_var + np.finfo(float).eps))
                output_tensor = np.multiply(self.weights, self.input_tensor_normalized) + self.bias
        
        if reformatted:
            # from 2d back to original 4d
            output_tensor = self.reformat(output_tensor)
        return output_tensor
        
            
    
    def backward(self, error_tensor):
        reformatted = False

        if error_tensor.ndim == 4:
            error_tensor = self.reformat(error_tensor)
            reformatted = True

        self.gradient_bias = np.sum(error_tensor, axis=0)
        self.gradient_weights = np.sum(error_tensor * self.input_tensor_normalized, axis=0)
        
        # to match the batch size (error_tensor.shape[0])
        mean_broadcasted = np.repeat(self.mean, error_tensor.shape[0], axis=0)
        var_broadcasted = np.repeat(self.var, error_tensor.shape[0], axis=0)

        grad_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor,
                    self.weights, mean_broadcasted, var_broadcasted)
        
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        
        if reformatted:
            grad_input = self.reformat(grad_input)
        
        return grad_input
    

    def reformat(self, tensor):
        if tensor.ndim == 4:
            b, h, m, n = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(b * m * n, h)
        
        if tensor.ndim == 2 and len(self.original_input_shape) == 4:
            b, h, m, n = self.original_input_shape
            return tensor.reshape(b, m, n, h).transpose(0, 3, 1, 2)
        
        return tensor
    
    def initialize(self, weights_initializer = None, bias_initializer = None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer