import tensorflow as tf
import numpy as np
import os
import cv2

mnist = tf.keras.datasets.mnist
(training_set, test_set) = mnist.load_data()

x_train, y_train = training_set
x_test, y_test = test_set

x_train = x_train/255.0
x_test = x_test/255.0
'''
cam = cv2.VideoCapture(0)
ret_val, img = cam.read()
print(ret_val)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resize = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
test_data = np.array([resize])
print (test_data.shape)
cv2.imwrite("test.png", img)
'''
resize = cv2.imread('test_8.png', 0)
test_data = np.array([resize])


model_list = [
    tf.keras.layers.Flatten(input_shape =(28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ]
model = tf.keras.models.Sequential(model_list)

checkpoint_dir = './training_checkpoints'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
results = probability_model(test_data)
print(np.argmax(results))


