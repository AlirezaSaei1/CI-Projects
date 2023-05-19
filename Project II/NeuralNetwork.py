import math
import numpy as np
import numba

class MLP():
  def __init__(self, input_size, output_size, layer_sizes, activations, loss_function="cross_entropy", learning_rate=0.001):
    self.input_size = input_size
    self.output_size = output_size
    self.layer_sizes = layer_sizes
    self.activations = activations
    self.loss_function = loss_function
    self.learning_rate = learning_rate
    self.layers = []

    prev_layer_size = self.input_size

    for layer_size, activation in zip(self.layer_sizes, self.activations):
        self.layers.append(Dense(prev_layer_size, layer_size))
        if activation == "Sigmoid":
            self.layers.append(Sigmoid())
        elif activation == "ReLU":
            self.layers.append(ReLU())
        else:
            raise ValueError("Invalid activation function")
        prev_layer_size = layer_size

    self.layers.append(Dense(prev_layer_size, self.output_size))
    if self.loss_function == "cross_entropy":
        self.loss = CategoricalCrossEntropyLoss()
        self.layers.append(Softmax())
    else:
        raise ValueError("Invalid loss function")

    self.optimizer = SGD(self.learning_rate)

  
  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X


  def backward(self, y_hat, y):
    gradient = self.loss.backward(y_hat, y)
    for layer in reversed(self.layers):
      gradient = layer.backward(gradient)
    return gradient

  
  def train(self, X, y, num_epochs):
    for epoch in range(num_epochs):
      y_hat = self.forward(X)
      loss = self.loss.forward(y_hat, y)
      gradient = self.backward(y_hat, y)
      for layer in self.layers:
        if isinstance(layer, Dense):
          self.optimizer.update(layer)
      print(f'Epoch {epoch+1} / {num_epochs}, Loss: {loss}')

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, b_input):
        self.weights_gradient = np.dot(self.inputs.T, b_input)
        self.biases_gradient = np.sum(b_input, axis=0, keepdims=True)
        b_output = np.dot(b_input, self.weights.T)
        return b_output


class Sigmoid:
    @staticmethod
    @numba.njit(fastmath=True)
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, inputs):
        self.outputs = self.sigmoid(inputs)
        return self.outputs

    def backward(self, b_input):
        output_gradients = self.outputs * (1 - self.outputs) * b_input
        return output_gradients


class ReLU:
    @staticmethod
    @numba.njit(fastmath=True)
    def relu(x):
        return np.maximum(0, x)

    def forward(self, inputs):
        self.outputs = self.relu(inputs)
        return self.outputs

    def backward(self, b_input):
        output_gradients = np.where(self.outputs > 0, b_input, 0)
        return output_gradients


class Softmax:
    @staticmethod
    @numba.njit(fastmath=True)
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, inputs):
        self.outputs = self.softmax(inputs)
        return self.outputs

    def backward(self, b_input):
        softmax_outputs = self.outputs

        jacobian = np.zeros((softmax_outputs.shape[0], softmax_outputs.shape[1], softmax_outputs.shape[1]))
        diag_indices = np.diag_indices(jacobian.shape[-1])
        jacobian[:, diag_indices[0], diag_indices[1]] = softmax_outputs * (1 - softmax_outputs)
        off_diag_indices = np.where(jacobian == 0)
        jacobian[:, off_diag_indices[0], off_diag_indices[1]] = -softmax_outputs.reshape(-1,1) * softmax_outputs

        dL_dInputs = np.dot(b_input, jacobian.T)
        return dL_dInputs


class CategoricalCrossEntropyLoss:
    @staticmethod
    @numba.njit(fastmath=True)
    def forward(softmax_output, class_label):
        return -math.log(softmax_output[class_label])

    @staticmethod
    @numba.njit(fastmath=True)
    def backward(softmax_output, class_label):
        gradient = np.zeros_like(softmax_output)
        gradient[class_label] = -1 / softmax_output[class_label]
        gradient[gradient == 0] = -softmax_output[gradient == 0] * softmax_output[class_label]
        gradient /= softmax_output[class_label]
        return gradient


class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, layer: Dense):
        layer.weights -= self.learning_rate * layer.weights_gradient
        layer.biases -= self.learning_rate * layer.biases_gradient
