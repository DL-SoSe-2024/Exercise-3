from Layers import Base
import numpy as np


class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = []

    def forward(self, input_tensor):
        activation = np.tanh(input_tensor)
        self.activations.append(activation)
        return activation

    def backward(self, error_tensor):
        return error_tensor * (1 - np.square(self.activations.pop()))