import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def compute_loss(y, y_pred):
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / np.size(y)

class ANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.rand(input_size, hidden_size) - 0.5
        self.W2 = np.random.rand(hidden_size, output_size) - 0.5
        self.B1 = np.zeros((1, hidden_size))
        self.B2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.B1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.B2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.A1)
        self.W2 += np.dot(self.A1.T, output_delta) * self.learning_rate
        self.B2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += np.dot(X.T, hidden_delta) * self.learning_rate
        self.B1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = compute_loss(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = ANN(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    nn.train(X, y, epochs=10000)
    for x in X:
        predicted = nn.forward(x.reshape(1, -1))
        print(f"Input: {x} -> Predicted Output: {predicted}")
