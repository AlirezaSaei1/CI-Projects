from keras.datasets import cifar10
from classification_models.keras import Classifiers

# uncomment this to download the weights of resnet34 if you don't have the .h5 file (85MB)
# import urllib.request
# url = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5'
# urllib.request.urlretrieve(url, 'resnet34_imagenet_1000.h5')


# Get the CIFAR-10 dataset
(x_train_subset, y_train_subset), (x_test_subset, y_test_subset) = cifar10.load_data()


# Load the pre-trained ResNet34 model
ResNet34, preprocess_input = Classifiers.get('resnet34')
x_train = preprocess_input(x_train_subset)
x_test = preprocess_input(x_test_subset)
model = ResNet34(input_shape=(32, 32, 3), weights='resnet34_imagenet_1000.h5')


# Extract features
train_features = model.predict(x_train)
test_features = model.predict(x_test)


print('Train Features shape:', train_features)
print('Test Features shape:', test_features.shape)

