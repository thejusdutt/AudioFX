import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from scipy.fftpack import fft, ifft

train_input = pd.read_csv("TrainInput.csv")
train_output = pd.read_csv("TrainOutput.csv")
test_input = pd.read_csv("TestInput.csv")
test_output = pd.read_csv("TestInput.csv")

train_input_fft = fft (train_input)
train_output_fft = fft (train_output)

test_input_fft = fft (test_input)
test_output_fft = fft (test_output)



model = keras.Sequential([

    keras.layers.Dense(12,input_dim=1,activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_input_fft, train_output_fft, epochs=5)

test_loss, test_acc = model.evaluate(test_input_fft,  test_output_fft, verbose=2)




""" model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



 """