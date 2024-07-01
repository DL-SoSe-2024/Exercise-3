from Layers import Base
import numpy as np

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = []

    def forward(self, input_tensor):
        activation = 1 / (1 + np.exp(-input_tensor))
        self.activations.append(activation)
        return activation

    def backward(self, error_tensor):
        activation = self.activations.pop()
        return error_tensor * activation * (1 - activation)