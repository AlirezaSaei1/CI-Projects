import numpy as np

class Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self,inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self,b_input):
        self.weights_gradient = np.dot(self.inputs.T, b_input)
        self.biases_gradient = np.sum(b_input, axis=0, keepdims=True)
        b_output = np.dot(b_input, self.weights.T)
        return b_output


class Sigmoid:
    def forward(self,inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self,b_input):
        derivative = self.outputs * (1 - self.outputs)
        return derivative * b_input


class ReLU:
    def forward(self,inputs):
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self,b_input):
        derivative = np.where(self.outputs > 0, 1, 0)
        return derivative * b_input


class Softmax:
    def forward(self,inputs):
        pass
        # // To do: Implement the softmax formula

    
    def backward(self,b_input):
        pass
        # // To do: Implement the softmax derivative with respect to the input


class Categorical_Cross_Entropy_loss:
    def forward(self,softmax_output,class_label):
        pass
        # // To do: Implement the CCE loss formula

    def backward(self,softmax_output,class_label):
        pass
        # // To do: Implement the CCE loss derivative with respect to predicted label


class SGD:
    def __init__(self,learning_rate = 0.001):
        self.learning_rate = learning_rate
        
    def update(self,layer):
        pass
        # // To do: Update layer params based on gradient descent rule