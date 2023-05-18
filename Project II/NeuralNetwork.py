import math
import numpy as np
import numba


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
