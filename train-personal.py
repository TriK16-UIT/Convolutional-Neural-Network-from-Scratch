import numpy as np
import os
import cv2
import keras.utils
from Network import Network
from Layer import *
from optimizer import GradientDescent, Adam
from sklearn.model_selection import train_test_split

data_length = 0
X = []
y = []
for i in range(0, 5):
    count = 0
    for k in os.listdir('./Personal/' + str(i) + '/'):
            img = cv2.imread('./Personal/' + str(i) + '/'+ k, cv2.IMREAD_GRAYSCALE)
            ## width, height
            img = cv2.resize(img, (28, 28))
            x_data = np.array(img)
            X.append(x_data)
            count = count + 1
    y_data = np.full((count, 1), i)
    y.append(y_data)
    data_length += count
X = np.array(X, dtype=np.float64)
y = np.array(y)
X = X.reshape(data_length, 1, 28, 28)
X = X.astype(np.float64) / 255
y = y.reshape(data_length, 1)
y = keras.utils.to_categorical(y)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
CNN = Network()
CNN.add_layers(Conv2D('Conv1', 2, 5, 1, (1, 28, 28), 'relu'))
CNN.add_layers(MaxPooling2D('Pool1', 2, 2))
CNN.add_layers(Conv2D('Conv2', 4, 3, 1, (2, 12, 12), 'sigmoid'))
CNN.add_layers(MaxPooling2D('Pool2', 2, 2))
CNN.add_layers(Dense('Dense1', 5*5*4, 5, activation='softmax'))
CNN.train(X_train, y_train, 10, Adam(0.001, 0.9, 0.999, 1e-8))
y_pred = CNN.predict(X_test)
acc = CNN.score(y_pred, y_test)
print(acc)
CNN.save_model('HandRegconition-Personal-epoch10')