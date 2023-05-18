import math

import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
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
    def forward(self, inputs):
        self.outputs = []
        for x in inputs:
            sig = 1 / (1 + math.exp(-x))
            self.outputs.append(sig)
        return self.outputs

    def backward(self, b_input):
        output_gradients = []
        for i, out in enumerate(self.outputs):
            grad = out * (1 - out) * b_input[i]
            output_gradients.append(grad)
        return output_gradients



class ReLU:
    def forward(self, inputs):
        self.outputs = []
        for x in inputs:
            relu = max(0, x)
            self.outputs.append(relu)
        return self.outputs

    def backward(self, b_input):
        output_gradients = []
        for i, out in enumerate(self.outputs):
            grad = b_input[i] if out > 0 else 0
            output_gradients.append(grad)
        return output_gradients


class Softmax:
    def forward(self, inputs):
        exp_inputs = []
        for x in inputs:
            exp_x = [math.exp(i - max(x)) for i in x]
            exp_inputs.append(exp_x)

        self.outputs = []
        for exp_x in exp_inputs:
            sum_exp_x = sum(exp_x)
            output = [x / sum_exp_x for x in exp_x]
            self.outputs.append(output)

        return self.outputs

    def backward(self,b_input):
        softmax_outputs = self.forward(b_input)

        jacobian = []
        for i in range(len(softmax_outputs)):
            row = []
            for j in range(len(softmax_outputs)):
                if i == j:
                    row.append(softmax_outputs[i] * (1 - softmax_outputs[i]))
                    # diagonal elements of jacobian are S_i * (1 - S_i)
                else:
                    row.append(-softmax_outputs[i] * softmax_outputs[j])
                    # off-diagonal elements are -S_i * S_j
            jacobian.append(row)

        # Calculate the derivative of loss with respect to inputs using chain rule and jacobian
        dL_dInputs = []
        for i in range(len(b_input)):
            row = []
            for j in range(len(b_input[0])):
                sum_jacob = 0
                for k in range(len(jacobian)):
                    sum_jacob += jacobian[k][j] * b_input[i][k]
                    # multiply jacobian of ith input with respect to all inputs with derivative of loss wrt ith input
                row.append(sum_jacob)
            dL_dInputs.append(row)

        return dL_dInputs


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