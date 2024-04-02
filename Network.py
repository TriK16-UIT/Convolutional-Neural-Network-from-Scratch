import numpy as np
import pickle
from Layer import *

##Cross entropy loss with respect to Softmax output
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15)) ## avoid log(0)
    
class Network:
    def __init__(self):
        self.layers = []
    
    def add_layers(self, layer):
        self.layers.append(layer)
    
    def forward(self, image):
        for layer in self.layers:
            image = layer.forward(image)
        return image
    
    def backward(self, gradient, optimizer):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, optimizer)
    
    def train(self, X_train, y_train, epochs, optimizer):
        for epoch in range(1, epochs + 1):
            print('\n--- Epoch {} ---'.format(epoch))
            avg_loss, num_corrects = 0, 0
            for i in range(len(X_train)):
                label = np.argmax(y_train[i])
                y_pred = self.forward(X_train[i])
                loss = cross_entropy_loss(y_train[i], y_pred)
                avg_loss += loss
                if (np.argmax(y_pred) == label):
                    num_corrects+=1
                #gradient is the derivative of the loss function
                gradient = y_pred - y_train[i]
                self.backward(gradient, optimizer)
            avg_loss = avg_loss / len(X_train)
            accuracy = (num_corrects / len(X_train)) * 100
            print(f"Loss: {avg_loss}, Accuracy: {accuracy}%")
            # self.save_model('HandRegconition')
    
    def predict(self, X_test):
        y = []
        for i in range(len(X_test)):
            y.append(self.forward(X_test[i]))
        return y
    
    def score(self, y_pred, y_true):
        num_corrects = 0
        for i in range(len(y_pred)):
            if np.argmax(y_pred[i]) == np.argmax(y_true[i]):
                num_corrects+=1
        accuracy = num_corrects/len(y_true) * 100
        return accuracy
    
    def save_model(self, path):
        model = [layer.get_config() for layer in self.layers]
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        network = Network()
        for layer in model:
            if (layer['type'] == 'Conv2D'):
                network.add_layers(Conv2D(initial=False))
            elif (layer['type'] == 'Dense'):
                network.add_layers(Dense(initial=False))
            elif (layer['type'] == 'MaxPooling2D'):
                network.add_layers(MaxPooling2D(initial=False))
            network.layers[-1].set_config(layer)
        return network
