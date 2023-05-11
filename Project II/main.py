from keras.datasets import cifar10
from classification_models.keras import Classifiers
from keras.activations import softmax
import numpy as np

# uncomment this to download the weights of resnet34 if you don't have the .h5 file (85MB)
# import urllib.request
# url = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5'
# urllib.request.urlretrieve(url, 'resnet34_imagenet_1000.h5')


# Get the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Categories (10): Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck


# Load the pre-trained ResNet34 model
ResNet34, preprocess_input = Classifiers.get('resnet34')
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
model = ResNet34(input_shape=(32, 32, 3), weights='resnet34_imagenet_1000.h5')


# Extract features
train_features = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=1, arr=model.predict(x_train))
test_features = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=1, arr=model.predict(x_test))


print('Train Features shape:', train_features.shape)
print('Test Features shape:', test_features.shape)

