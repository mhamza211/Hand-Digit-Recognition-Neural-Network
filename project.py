#import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data into data sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#flattened 2d array into 1d array
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)

#normalize the terms
x_train_flattened = x_train_flattened / 255
x_test_flattened = x_test_flattened / 255

#build a neural network model
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation='relu'),  # hidden layer
    keras.layers.Dense(10, activation='sigmoid')  # output layer
])

#compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#train the model
model.fit(x_train_flattened, y_train, epochs=5)

#visualize a test image
plt.matshow(x_test[1])
plt.show()

#predict using the trained model
y_predicted = model.predict(x_test_flattened)
print(np.argmax(y_predicted[1]))
