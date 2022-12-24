import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
Implementing a neural network using Keras
"""

# From Keras, load the MNIST digits classification dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 

# Visualize the first 10 instances (digits) from the dataset
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray') 
    plt.axis('off')  
plt.show()

# Verify the shape of the instances and associated label
print("In the training set, there are", x_train.shape, "instances (2D grayscale image data with 28×28 pixels. \
In turn, every image is represented as a 28×28 array rather than a 1D array of size 784. \
Pixel values range from 0 (white) to 255 (black).) \
The associated labels are digits ranging from 0 to 9.") 

# Scale the input feature down to 0-1 values
x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") /255.0 

# Create a Sequential model
model = keras.Sequential() 

# Build a first layer to the model, that will convert each 2D image into a 1D array. 
model.add(keras.layers.Flatten(input_shape= [28,28])) 


# 'Dense layer' with 300 neurons and the ReLU activation function. 
model.add(keras.layers.Dense(300, activation="relu", name="first_hidden_layer")) 


# 'Dense layer' with 100 neurons, also using the ReLU activation function.
model.add(keras.layers.Dense(100, activation="relu", name="second_hidden_layer")) 

# Build an output layer to the model.
# 'Dense layer' with 10 neurons (one per class), using the softmax activation function.
model.add(keras.layers.Dense(10, activation="softmax", name="output_layer")) 

model.summary()

# compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
# apply model on training instances
model.fit(x_train, y_train, epochs = 20) 


# test the model
plt.close('all')
y_pred = model.predict(x_test) 
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title('Predicted label: ' + str(np.argmax(y_pred[i]))) 
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')  
plt.show()


