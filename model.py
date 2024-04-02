from Network import Network
from Layer import *
from keras.datasets import mnist
from optimizer import GradientDescent, Adam
import keras.utils

def preprocess(X, y, limit=None):
    X = X[:limit, :]
    y = y[:limit]
    X = X.reshape(len(X), 1, 28, 28)
    X = X.astype(np.float64) / 255
    y = keras.utils.to_categorical(y, num_classes=10)

    return X, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess(x_train, y_train, 1000)
x_test, y_test = preprocess(x_test, y_test, 100)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

tri3 = Network()
tri3.add_layers(Conv2D('Conv1', 2, 5, 1, (1, 28, 28), 'relu'))
tri3.add_layers(MaxPooling2D('Pool1', 2, 2))
tri3.add_layers(Conv2D('Conv2', 4, 3, 1, (2, 12, 12), 'sigmoid'))
tri3.add_layers(MaxPooling2D('Pool2', 2, 2))
tri3.add_layers(Dense('Dense1', 5*5*4, 10, activation='softmax'))
tri3.train(x_train, y_train, 10, Adam(0.001, 0.9, 0.999, 1e-8))
# tri3.train(x_train, y_train, 5, GradientDescent(0.001))
y_pred = tri3.predict(x_test)
acc = tri3.score(y_pred, y_test)
print(acc)
tri3.save_model('test')

print("Try to load model...")

loaded_model = Network()
loaded_model = loaded_model.load_model('test')
y_pred = loaded_model.predict(x_test)
acc = loaded_model.score(y_pred, y_test)
print(acc)

