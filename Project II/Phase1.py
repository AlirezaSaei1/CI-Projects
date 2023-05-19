import torch
from torch import nn
from torch import optim
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import math
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(train_data, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnet34(pretrained=True)

modules = list(resnet_model.children())[:-1]
resnet_model = torch.nn.Sequential(*modules)

resnet_model.eval()

for param in resnet_model.parameters():
  param.requires_grad = False

resnet_model.to(device)

def extract_features(dataset):
  features = []
  labels = []
  with torch.no_grad():
    for images, label in dataset:
      images = images.to(device)
      output = resnet_model(images)
      features.append(output.cpu().numpy())
      labels.append(label.numpy())
  features = np.concatenate(features, axis=0)
  labels = np.concatenate(labels)
  return features, labels

x_train, y_train = extract_features(train_loader)
x_test, y_test = extract_features(test_loader)

x_train = x_train.reshape(50000, 512)
y_train = np.array(y_train)

def init_params():
    W1 = np.random.rand(20, 512) - 0.5
    b1 = np.random.rand(20, 1)  - 0.5
    W2 = np.random.rand(10, 20) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = 50000
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(f'Accuracy: {get_accuracy(predictions, Y) * 100} %')
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(x_train.T, y_train, 0.001, 1000)

predictions_train = get_predictions(forward_prop(W1, b1, W2, b2, x_train.T)[-1])
cm_train = confusion_matrix(y_train, predictions_train)
f1_train = f1_score(y_train, predictions_train, average='weighted')
print("Training confusion matrix:\n", cm_train)
print("Training F1 score:", f1_train)

predictions_test = get_predictions(forward_prop(W1, b1, W2, b2, x_test.T)[-1])
cm_test = confusion_matrix(y_test, predictions_test)
f1_test = f1_score(y_test, predictions_test, average='weighted')
print("\nTest confusion matrix:\n", cm_test)
print("Test F1 score:", f1_test)