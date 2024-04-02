class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

class Adam:
    def __init__ (self, learning_rate, beta1, beta2, epsilon):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon