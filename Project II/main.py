import tensorflow as tf
from tensorflow import keras

# Get the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

if __name__ == '__main__':
    print('Project 2 Init')