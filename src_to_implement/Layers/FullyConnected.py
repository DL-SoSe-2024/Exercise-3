from Layers import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)
        self.input_tensors = []
        
    def forward(self, input_tensor):
        if input_tensor.ndim == 1:
            input_tensor = np.hstack((input_tensor, np.ones((1))))
        else:    
            input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        self.input_tensors.append(input_tensor)
        return np.matmul(self.input_tensor, self.weights)
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights
    
    def backward(self, error_tensor):
        flag = 0
        input_tensor = self.input_tensors.pop()
        if input_tensor.ndim == 1:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        if error_tensor.ndim == 1:
            error_tensor = np.expand_dims(error_tensor, axis=0)
            flag = 1
        self.gradient_weights = np.matmul(np.transpose(input_tensor), error_tensor)
        if hasattr(self, "optimizer"):
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if flag:
            return np.delete(np.matmul(error_tensor, np.transpose(self.weights))[0,...], self.weights.shape[0] - 1, axis = 0)
        else:
            return np.delete(np.matmul(error_tensor, np.transpose(self.weights)), self.weights.shape[0] - 1, axis = 1)
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
        
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.weights = np.vstack((self.weights, bias_initializer.initialize((1, self.output_size), 1, self.output_size)))
        
    @property
    def input_tensor(self):
        return self.input_tensors[-1]