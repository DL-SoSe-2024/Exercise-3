import copy
from Optimization import *

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.phase = "train"
        
    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, phase):
        self._phase = phase
        if phase == "test":
            for layer in self.layers:
                layer.testing_phase = True
        elif phase == "train":
            for layer in self.layers:
                layer.testing_phase = False
        
    
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
        
    def forward(self):
        reg_loss = 0
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer is not None and layer.trainable:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)
        self.loss_layer = Loss.CrossEntropyLoss()
        output = self.loss_layer.forward(input_tensor, self.label_tensor) + reg_loss
        return output
        
    def backward(self):
        if self.label_tensor is not None:
            error_tensor = self.loss_layer.backward(self.label_tensor)
            for layer in reversed(self.layers):
                error_tensor = layer.backward(error_tensor)
            self.label_tensor = None
        else:
            raise RuntimeError("Do a forward pass first")
    
    def train(self, iterations):
        self.phase = "train"
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()
    
    def test(self, input_tensor):
        self.phase = "test"
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
