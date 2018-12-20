import numpy as np;


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


@np.vectorize
def d_sigmoid(x):
    sig = 1 / (1 + np.e ** -x)
    return sig * (1 - sig)


@np.vectorize
def tanh(x):
    return np.tanh(x)


activation_function = sigmoid


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes)
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes)

        self.bias_h = np.random.rand(self.hidden_nodes, 1)
        self.bias_o = np.random.rand(self.output_nodes, 1)

    def train(self, input_array, target_array):
        inputs = np.array(input_array, ndmin=2).T
        hidden = np.dot(self.weights_ih, inputs)
        hidden = hidden + self.bias_h
        hidden = activation_function(hidden)

        outputs = np.dot(self.weights_ho, hidden)
        outputs = outputs + self.bias_o
        outputs = activation_function(outputs)

        targets = np.array(target_array, ndmin=2).T
        output_errors = targets - outputs

        # Calculate Gradient
        gradients = output_errors * d_sigmoid(outputs)
        gradients = self.learning_rate * gradients

        # Calculate deltas
        hidden_t = np.array(hidden).T

        weight_ho_deltas = np.dot(gradients, hidden_t)

        # Adjust weights by delta
        self.weights_ho += weight_ho_deltas
        self.bias_o += gradients

        # Calculate hidden layer errors
        who_t = np.array(self.weights_ho).T
        hidden_errors = np.dot(who_t, output_errors)

        # Calculate hidden gradient
        hidden_gradient = d_sigmoid(hidden)
        hidden_gradient = self.learning_rate * hidden_errors * hidden_gradient
        input_t = np.array(inputs).T
        weight_ih_deltas = np.dot(hidden_gradient, input_t)

        self.weights_ih += weight_ih_deltas
        self.bias_h += hidden_gradient

    def predict(self, input_vector):
        # Generating the Hidden Outputs
        inputs = np.array(input_vector, ndmin=2).T
        hidden = np.dot(self.weights_ih, inputs)

        # activation function
        hidden = activation_function(hidden)

        # Generating output
        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = activation_function(output)

        return output
