import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1.0 - s)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.t = 0
    def forward(self, image):
        pass
    def backward(self, output_gradient, optimizer):
        pass
    def get_config(self):
        pass
    def set_config(self, config):
        pass
    def updateWeightswithAdam(self, weights, gradients, m, v, learning_rate, beta1, beta2, epsilon):
        self.t += 1
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)

        m_corr = m / (1 - beta1 ** self.t)
        v_corr = v / (1 - beta2 ** self.t)

        weights -= learning_rate * m_corr / (np.sqrt(v_corr) + epsilon)
        return weights, m, v

class Conv2D(Layer):
    def __init__(self, name = None, num_kernels = None, kernel_size = None, stride = None, input_dimension = None, activation = None, initial = True):
        super().__init__()
        if (initial == False):
            return
        ## Input dimension is D, H, W
        self.name = name
        self.stride = stride
        self.kernel_size = kernel_size
        self.depth = num_kernels
        if (activation == None):
            self.kernels = np.random.randn(self.depth, input_dimension[0], kernel_size, kernel_size) * 0.1
        elif (activation == 'relu' or activation == 'leaky_relu'):
            self.kernels = np.random.randn(self.depth, input_dimension[0], kernel_size, kernel_size) * np.sqrt(2. / (kernel_size * kernel_size * self.depth))
        elif (activation == 'sigmoid'):
            input_nodes = kernel_size * kernel_size * input_dimension[0]
            output_nodes = kernel_size * kernel_size * self.depth
            limit = np.sqrt(6 / (input_nodes + output_nodes))
            self.kernels = np.random.uniform(-limit, limit, (self.depth, input_dimension[0], kernel_size, kernel_size))
        self.biases = np.zeros(self.depth)
        self.activation = activation
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        self.m_kernels = np.zeros_like(self.kernels)
        self.v_kernels = np.zeros_like(self.kernels)

    def forward(self, image):
        self.input = image
        input_dimension = image.shape
        self.output = np.zeros((self.depth, (input_dimension[1] - self.kernel_size) // self.stride + 1, (input_dimension[2] - self.kernel_size) // self.stride + 1))
        for kernel in range(self.depth):
            flag_row = output_row = 0
            while flag_row + self.kernel_size <= input_dimension[1]:
                flag_col = output_col = 0
                while flag_col + self.kernel_size <= input_dimension[2]:
                    roi = image[:, flag_row:flag_row + self.kernel_size, flag_col:flag_col + self.kernel_size]
                    self.output[kernel, output_row, output_col] += np.sum(self.kernels[kernel] * roi)
                    flag_col += self.stride
                    output_col += 1
                self.output[kernel] += self.biases[kernel]
                flag_row += self.stride
                output_row += 1
        self.last_output = self.output
        if (self.activation == 'relu'):
            self.output = relu(self.output)
        if (self.activation == 'leaky_relu'):
            self.output = leaky_relu(self.output, 0.001)
        if (self.activation == 'sigmoid'):
            self.output = sigmoid(self.output)
        return self.output
    
    def backward(self, output_gradient, optimizer):
        input_gradient = np.zeros(self.input.shape)
        kernels_gradient = np.zeros(self.kernels.shape)
        biases_gradient = np.zeros(self.depth)
        input_dimension = self.input.shape
        if (self.activation == 'relu'):
            output_gradient = output_gradient * relu_derivative(self.last_output)
        if (self.activation == 'leaky_relu'):
            output_gradient = output_gradient * leaky_relu_derivative(self.last_output, 0.01)
        if (self.activation == 'sigmoid'):
            output_gradient = output_gradient * sigmoid_derivative(self.last_output)

        for kernel in range(self.depth):
            flag_row = output_row = 0
            while flag_row + self.kernel_size <= input_dimension[1]:
                flag_col = output_col = 0
                while flag_col + self.kernel_size <= input_dimension[2]:
                    roi = self.input[:, flag_row:flag_row + self.kernel_size, flag_col:flag_col + self.kernel_size]
                    kernels_gradient[kernel] += np.sum(output_gradient[kernel, output_row, output_col] * roi, axis=0)
                    input_gradient[:, flag_row:flag_row + self.kernel_size, flag_col:flag_col + self.kernel_size] += output_gradient[kernel, output_row, output_col] * self.kernels[kernel]
                    flag_col += self.stride
                    output_col += 1
                biases_gradient[kernel] += np.sum(output_gradient[kernel])
                flag_row += self.stride
                output_row += 1
        if (optimizer.__class__.__name__ == 'GradientDescent'):
            self.kernels -= optimizer.learning_rate * kernels_gradient
            for i in range(self.depth):
                self.biases[i] -= optimizer.learning_rate * biases_gradient[i]
        elif (optimizer.__class__.__name__ == 'Adam'):
            self.biases, self.m_biases, self.v_biases = self.updateWeightswithAdam(self.biases, biases_gradient, self.m_biases, self.v_biases, optimizer.learning_rate, optimizer.beta1, optimizer.beta2, optimizer.epsilon)
            self.kernels, self.m_kernels, self.v_kernels = self.updateWeightswithAdam(self.kernels, kernels_gradient, self.m_kernels, self.v_kernels, optimizer.learning_rate, optimizer.beta1, optimizer.beta2, optimizer.epsilon)
        return input_gradient

    def get_config(self):
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'stride': self.stride,
            'kernel_size': self.kernel_size,
            'depth': self.depth,
            'kernels': self.kernels,
            'biases': self.biases,
            'activation': self.activation,
            'm_biases': self.m_biases,
            'v_biases': self.v_biases,
            'm_kernels': self.m_kernels,
            'v_kernels': self.v_kernels,
            't': self.t
        }
    
    def set_config(self, config):
        self.name = config['name']
        self.stride = config['stride']
        self.kernel_size = config['kernel_size']
        self.depth = config['depth']
        self.kernels = config['kernels']
        self.biases = config['biases']
        self.activation = config['activation']
        self.m_biases = config['m_biases']
        self.v_biases = config['v_biases']
        self.m_kernels = config['m_kernels']
        self.v_kernels = config['v_kernels']
        self.t = config['t']

class Dense(Layer):
    def __init__(self, name = None, input_size = None, output_size = None, activation = None, initial = True):
        super().__init__()
        if (initial == False):
            return
        self.name = name
        self.biases = np.zeros(output_size)
        self.output_size = output_size
        self.activation = activation
        self.input_dimension = None
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)

    def forward(self, image):
        self.input_dimension = image.shape
        self.input = image.flatten()
        self.output = np.dot(self.input, self.weights) + self.biases

        return softmax(self.output)
    
    def backward(self, output_gradient, optimizer):        
        weights_gradient = np.dot(output_gradient[:, np.newaxis], self.input[:, np.newaxis].T)
        input_gradient = np.dot(self.weights, output_gradient[:, np.newaxis])
        if (optimizer.__class__.__name__ == 'GradientDescent'):
            self.weights -= optimizer.learning_rate * weights_gradient.T
            self.biases -= optimizer.learning_rate * output_gradient
        elif (optimizer.__class__.__name__ == 'Adam'):
            self.weights, self.m_weights, self.v_weights = self.updateWeightswithAdam(self.weights, weights_gradient.T, self.m_weights, self.v_weights, optimizer.learning_rate, optimizer.beta1, optimizer.beta2, optimizer.epsilon)
            self.biases, self.m_biases, self.v_biases = self.updateWeightswithAdam(self.biases, output_gradient, self.m_biases, self.v_biases, optimizer.learning_rate, optimizer.beta1, optimizer.beta2, optimizer.epsilon)
        return input_gradient.reshape(self.input_dimension)
    
    def get_config(self):
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'biases': self.biases,
            'activation': self.activation,
            'weights': self.weights,
            'm_biases': self.m_biases,
            'v_biases': self.v_biases,
            'm_weights': self.m_weights,
            'v_weights': self.v_weights,
            't': self.t
        }
    
    def set_config(self, config):
        self.name = config['name']
        self.biases = config['biases']
        self.activation = config['activation']
        self.weights = config['weights']
        self.m_biases = config['m_biases']
        self.v_biases = config['v_biases']
        self.m_weights = config['m_weights']
        self.v_weights = config['v_weights']
        self.t = config['t']

class MaxPooling2D(Layer):
    def __init__(self, name = None, stride = None, size = None, initial = True):
        super().__init__()
        if (initial == False):
            return
        self.name = name
        self.stride = stride
        self.size = size

    def forward(self, image):
        self.input = image
        channels, height, width = self.input.shape
        updated_height = int((height - self.size) / self.stride) + 1
        updated_width = int((width - self.size) / self.stride) + 1
        self.output = np.zeros((channels, updated_height, updated_width))

        for channel in range(channels):
            flag_row = output_row = 0
            while flag_row + self.size <= height:
                flag_col = output_col = 0
                while flag_col + self.size <= width:
                    roi = image[channel, flag_row:flag_row + self.size, flag_col:flag_col + self.size]
                    self.output[channel, output_row, output_col] = np.max(roi)
                    flag_col += self.stride
                    output_col += 1
                flag_row += self.stride
                output_row += 1
        return self.output
    
    def backward(self, output_gradient, optimizer):
        channels, height, width = self.input.shape
        input_gradient = np.zeros(self.input.shape)

        for channel in range(channels):
            flag_row = output_row = 0
            while flag_row + self.size <= height:
                flag_col = output_col = 0
                while flag_col + self.size <= width:
                    roi = self.input[channel, flag_row:flag_row + self.size, flag_col:flag_col + self.size]
                    (row, col) = np.unravel_index(np.nanargmax(roi), roi.shape)
                    input_gradient[channel, flag_row + row, flag_col + col] += output_gradient[channel, output_row, output_col]
                    flag_col += self.stride
                    output_col += 1
                flag_row += self.stride
                output_row += 1
        return input_gradient

    def get_config(self):
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'stride': self.stride,
            'size': self.size
        }
    
    def set_config(self, config):
        self.name = config['name']
        self.stride = config['stride']
        self.size = config['size']