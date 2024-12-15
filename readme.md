# Neural Network Implementation

This project implements a simple artificial neural network (ANN) from scratch using Python and NumPy. The neural network is trained to solve the XOR problem, which is a classic example of a problem that is not linearly separable.

## Features
- A feedforward neural network with one hidden layer.
- Sigmoid activation function for non-linear transformations.
- Binary Cross-Entropy loss function.
- Backpropagation for weight and bias updates.

## Requirements

- Python 3.7+
- NumPy

Install dependencies using pip:
```bash
pip install numpy 
```

## Code Structure

### Functions
- `sigmoid(x)`: Implements the sigmoid activation function.
- `sigmoid_derivative(x)`: Calculates the derivative of the sigmoid function.
- `compute_loss(y, y_pred)`: Computes the Binary Cross-Entropy loss.

### ANN Class
The `ANN` class implements the neural network. It includes:

- `__init__(self, input_size, hidden_size, output_size, learning_rate)`: Initializes weights and biases for the network.
- `forward(self, X)`: Performs the forward pass to calculate predictions.
- `backward(self, X, y, output)`: Implements backpropagation to compute errors and update weights and biases.
- `train(self, X, y, epochs)`: Trains the neural network for a specified number of epochs.

## How to Run
1. Define the input and output data for the XOR problem:
    ```python
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    ```

2. Create an instance of the `ANN` class:
    ```python
    nn = ANN(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    ```

3. Train the neural network:
    ```python
    nn.train(X, y, epochs=10000)
    ```

4. Test the neural network:
    ```python
    for x in X:
        predicted = nn.forward(x.reshape(1, -1))
        print(f"Input: {x} -> Predicted Output: {predicted}")
    ```

## Example Output
After training, the network should predict outputs close to the target values for the XOR problem:

```
Input: [0 0] -> Predicted Output: [[0.01]]
Input: [0 1] -> Predicted Output: [[0.98]]
Input: [1 0] -> Predicted Output: [[0.98]]
Input: [1 1] -> Predicted Output: [[0.02]]
```

## Limitations
- This implementation is limited to small datasets and simple problems due to the use of basic matrix operations and sigmoid activation.
- The model is not optimized for large-scale computations.

## License
This project is open-source.

