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
        self.moving_var = None
        self.alpha = 0.8
    
    def forward(self, input_tensor):
        if input_tensor.ndim == 2:
            if not self.testing_phase:
                mean = np.repeat(np.mean(input_tensor, axis = 0, dtype=float, keepdims=True), [input_tensor.shape[0],], axis=0)
                var = np.repeat(np.var(input_tensor, axis = 0, dtype=float, keepdims=True), [input_tensor.shape[0],], axis=0)
                if self.moving_mean is None:
                    self.moving_mean = (1-self.alpha) * mean
                    self.moving_var = (1-self.alpha) * var
                else:
                    self.moving_mean = self.alpha * self.moving_mean + (1-self.alpha) * mean
                    self.moving_var = self.alpha * self.moving_var + (1-self.alpha) * var
                input_tensor_normalized = np.divide((input_tensor - mean), np.sqrt(var + np.finfo.eps(float)))    
            else:
                input_tensor_normalized = np.divide((input_tensor - self.moving_mean), np.sqrt(self.moving_var + np.finfo.eps(float)))
            return np.multiply(self.weights, input_tensor_normalized) + self.bias
            
    
    def backward(self, error_tensor):
        grad_bias = np.sum(error_tensor, axis=0)
        gradient_weights = np.sum(error_tensor * self.weights, axis=0)

        grad_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor,
                    self.weights, self.mean, self.var)
        
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, grad_bias)
    
    def initialize(self, **args):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer