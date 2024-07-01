import numpy as np
from Layers import Base, FullyConnected, TanH, Sigmoid

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.memorize = False
        self.trainable = True
        self.input_gate_layer = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.input_gate_activation = TanH.TanH()
        self.output_gate_layer = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.output_gate_activation = Sigmoid.Sigmoid()
            
    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize
    
    def forward(self, input_tensor):
        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)
        for t in range(input_tensor.shape[0]):
            input_vector = input_tensor[t,...]
            #print(input_vector.shape, self.hidden_state.shape)
            x_tilda = np.concatenate([input_vector, self.hidden_state])
            x = self.input_gate_layer.forward(x_tilda)
            self.hidden_state = self.input_gate_activation.forward(x + self.calculate_regularization_loss(self.input_gate_layer))
            output_tensor[t,...] = self.output_gate_activation.forward(self.output_gate_layer.forward(self.hidden_state) + self.calculate_regularization_loss(self.output_gate_layer))
        return output_tensor
    
    def backward(self, error_tensor):
        hidden_error_vector = np.zeros((self.hidden_size))
        error_tensor_input = np.zeros((error_tensor.shape[0], self.input_size))
        output_gate_update = np.zeros_like(self.output_gate_layer.weights)
        self.gradient_weights = np.zeros_like(self.input_gate_layer.weights)
        for t in reversed(range(error_tensor.shape[0])):
            error_vector = error_tensor[t,...]
            intermediate_error_vector = self.output_gate_layer.backward(self.output_gate_activation.backward(error_vector)) + hidden_error_vector
            output_gate_update += intermediate_error_vector
            input_error_vector_tilda = self.input_gate_layer.backward(self.input_gate_activation.backward(intermediate_error_vector))
            self.gradient_weights += self.input_gate_layer.gradient_weights
            hidden_error_vector = input_error_vector_tilda[-self.hidden_size:]
            error_tensor_input[t,...] = input_error_vector_tilda[:self.input_size]
        if hasattr(self, "optimizer"):
            self.output_gate_layer.weights = self.optimizer.calculate_update(self.output_gate_layer.weights, output_gate_update)
            self.input_gate_layer.weights = self.optimizer.calculate_update(self.input_gate_layer.weights, self.gradient_weights)
        return error_tensor_input
        
    def initialize(self, weights_initializer, bias_initializer):
        self.weights_intializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.input_gate_layer.initialize(self.weights_intializer, self.bias_initializer)
        self.output_gate_layer.initialize(self.weights_intializer, self.bias_initializer)

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def weights(self):
        return self.input_gate_layer.weights
    
    @weights.setter
    def weights(self, weights):
        self.input_gate_layer.weights = weights
        
    @property
    def gradient_weights(self):
        return self.input_gate_layer.gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.input_gate_layer.gradient_weights = gradient_weights
    
    def calculate_regularization_loss(self, layer):
        if hasattr(self, "optimizer") and self.optimizer.regularizer is not None:
            return self.optimizer.regularizer.norm(layer.weights)
        else:
            return 0