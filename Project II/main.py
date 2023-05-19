from keras.datasets import cifar10
from classification_models.keras import Classifiers
import numpy as np
from keras.models import Model
from NeuralNetwork import Dense, ReLU, Softmax, CategoricalCrossEntropyLoss, SGD


# Get the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Categories (10): Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck


ResNet34, preprocess_input = Classifiers.get('resnet34')
base_model = ResNet34(input_shape=(32, 32, 3), weights='resnet34_imagenet_1000.h5')

# Remove last layer
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Preprocess input data
x_train_features = preprocess_input(x_train)
x_test_features = preprocess_input(x_test)

# Extract features
train_features = model.predict(x_train_features)
test_features = model.predict(x_test_features)
print(train_features)
print('Train Features shape:', train_features.shape)
print('Test Features shape:', test_features.shape)

layers = [
    Dense(512, 20),
    ReLU(),
    Dense(20, 10),
    Softmax()
]
CCE_loss = CategoricalCrossEntropyLoss()
opt = SGD()

for epoch in range(100):
    print(f'Epoch: {epoch}')
    loss = 0.0
    accuracy = 0.0
    for x in range(len(x_train)):
        # forward
        layers[0].forward(train_features[x])
        layers[1].forward(layers[0].output)
        layers[2].forward(layers[1].output)
        layers[3].forward(layers[2].output)
        loss += CCE_loss.forward(layers[3].output, y_train[x])

        y_predict = np.argmax(layers[3].output, axis=1)
        accuracy += np.mean(y_train[x] == y_predict)

        # backward
        CCE_loss.backward(layers[3].output, y_train[x])
        layers[3].backward(CCE_loss.b_output)
        layers[2].backward(layers[3].b_output)
        layers[1].backward(layers[2].b_output)
        layers[0].backward(layers[1].b_output)

        opt.update(layers[0])
        opt.update(layers[1])

    loss /= len(x_train)
    accuracy /= len(x_train)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
