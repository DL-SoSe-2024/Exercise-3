import numpy as np
from Layers import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.output_tensor = None

    def forward(self, input_tensor):
        input_tensor = np.exp(input_tensor - np.amax(input_tensor, axis=1, keepdims = True))
        self.output_tensor =  input_tensor/input_tensor.sum(axis = 1, keepdims = True)
        return self.output_tensor
        
    def backward(self, label_tensor):
        return np.multiply(self.output_tensor, label_tensor - np.sum(np.multiply(label_tensor, self.output_tensor), axis = 1, keepdims= True))