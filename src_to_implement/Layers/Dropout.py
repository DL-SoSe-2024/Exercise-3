import numpy as np
from Layers import Base
import time

class Dropout(Base.BaseLayer):
    def __init__(self, probability) -> None:
        super().__init__()
        self.probability = probability
        self.mask = None
        
    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        self.mask = []
        dropped_tensor = np.zeros_like(input_tensor)
        np.random.seed(int(time.time()))
        for i in range(input_tensor.shape[0]):
            batch_mask = np.random.choice([0, 1], input_tensor.shape[1], p = [1-self.probability, self.probability]) 
            dropped_tensor[i,...] = np.multiply(input_tensor[i,...], batch_mask)
            self.mask.append(batch_mask)
        self.mask = np.stack(self.mask, axis = 0)
        return (1/self.probability) * dropped_tensor
    
    def backward(self, error_tensor):
        dropped_tensor = np.zeros_like(error_tensor)
        for i in range(error_tensor.shape[0]):
            dropped_tensor[i,...] = np.multiply(error_tensor[i,...], self.mask[i,...])
        return (1/self.probability) * dropped_tensor
            