import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor_epsilon = None
        
    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor_epsilon = np.add(prediction_tensor,np.finfo(float).eps)
        return -1 * np.sum(np.multiply(label_tensor, np.log(self.prediction_tensor_epsilon)))
    
    def backward(self, label_tensor):
        return -1 * np.divide(label_tensor, self.prediction_tensor_epsilon)