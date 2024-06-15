import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0.0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v

class Adam:
    def __init__(self, learning_rate, mu, rho) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.v = 0.0
        self.r = 0.0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)
        v_dash = self.v/(1 - self.mu ** self.k)
        r_dash = self.r/(1 - self.rho ** self.k)
        self.k += 1
        return weight_tensor - self.learning_rate * np.divide(v_dash, np.add(np.sqrt(r_dash), np.finfo(float).eps))