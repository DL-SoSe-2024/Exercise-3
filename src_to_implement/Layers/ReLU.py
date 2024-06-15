import numpy as np
from Layers import Base


class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        return np.maximum(0, np.sign(self.input_tensor)) * error_tensor


