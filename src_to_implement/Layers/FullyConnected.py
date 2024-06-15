from Layers import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)
        self.input_tensor = None
        
    def forward(self, input_tensor):
        self.input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        return np.matmul(self.input_tensor, self.weights)
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    def backward(self, error_tensor):
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor)
        if hasattr(self, "optimizer"):
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
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