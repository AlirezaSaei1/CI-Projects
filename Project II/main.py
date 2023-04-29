import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet34, preprocess_input
from tensorflow.keras.datasets import cifar10

# Get the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# Load the pre-trained ResNet34 model
resnet = ResNet34(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
resnet.trainable = False


if __name__ == '__main__':
    print('Project 2 Init')