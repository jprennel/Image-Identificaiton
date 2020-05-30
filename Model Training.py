# Import libraries

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Dataset of 28x28 pixel images of hand-written digits 0-9
mnist = tf.keras.datasets.mnist

# one set to train the model, one  set to validate the training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize each pixel from 0-255 to 0-1 (easier for network to learn
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Sequential model is good for single-input, single-output stacks of layers
model = tf.keras.models.Sequential()

# Input Layer
model.add(tf.keras.layers.Flatten())

# Two hidden layers - 128 neurons per layer, Rectified Linear Unit Activation Function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Output layer - 10 possible outcomes (a number between 0-9) / softmax gives probabilistic output
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

# Minimize loss to improve accuracy using the adam optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=4)

# Calculate Validation Loss & Accuracy (Expect the model accuracy to be slightly lower than training)
val_loss, val_acc = model.evaluate(x_test, y_test)

# Use the model to predict the values of the images in the x_test dataset
predictions = model.predict(x_test)

# For the first image in the set, show the predicted value (0-9)
print(f'\nPredicted image value: {np.argmax(predictions[0])}')

# Show the probability array for the image (notice that eighth element has a probability closest to 1)
print(f'\nProbability array for ', predictions[0])

# Show a sample image
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()

