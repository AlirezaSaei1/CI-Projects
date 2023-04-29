import numpy as np

class NeuralNetwork:
    def __init__(self):
        pass


class Dense:
    def __init__(self,n_inputs,n_neurons):
        pass
        # // To do: Define initial weight and bias
    
    def forward(self,inputs):
        pass
        # // To do: Define input and output

    def backward(self,b_input):
        pass
        # // To do: Weight and bias gradients


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