"""
    Steps to build a neural network:    

    Initialize the weights and biases for the network

    Create the activation and loss functions (sigmoid, ReLU, softmax, MSE)

    Implement the forward pass

    Create the derivatives of the activation functions and loss functions

    Implement the backward pass

    Update the weights and biases

    Train the network

    Test the network

"""
"""
    Notes for self:
        x_(i+1) = W_i dot x_i + b_i
        For dot product, the number of columns in the first matrix must be equal to the number of rows in the second matrix

        If input is 5x1, the weight matrix should be 10x5, and the bias matrix should be 10x1
        Therefore the dimensions of the weight matrix should be (n, m)  (next, current)
        where n is the number of nodes in the next layer and m is the number of nodes in the current layer
        The dimensions of the bias matrix should be (n, 1)              (next, 1)

        |'''''|    This        |'''''|
        |This |    weight ===> |Next |
        |Layer|    & bias      |Layer|
        |     |                |     |
        |     |===========>    |     |
        |     |                |     |
        |,,,,,|                |,,,,,|

"""

import numpy as np

def sigmoid(Z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-Z))

def sigmoidDerivative(Z: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid activation function"""
    # return sigmoid(Z) * (1 - sigmoid(Z))
    return Z * (1 - Z)

def relu(Z: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(0, Z)

def reluDerivative(Z: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU activation function"""
    return np.where(Z > 0, 1, 0)

# def softmax(Z: np.ndarray) -> np.ndarray:
#     """Softmax activation function"""
#     exp_Z = np.exp(Z - np.max(Z))  # For numerical stability
#     return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# def softmax_derivative(Z: np.ndarray) -> np.ndarray:
#     """Derivative of the softmax activation function"""
#     # Derivative of softmax is more complex. For simplicity, assume cross-entropy loss
#     return Z * (1 - Z)  # This is only an approximation

def activationFunction(func: str, Z: np.ndarray) -> np.ndarray:
    """Return the result of the activation function given the function name and input"""
    if func == "sigmoid":
        return sigmoid(Z)
    elif func == "relu":
        return relu(Z)
    # elif func == "softmax":
    #     return softmax(Z)
    raise Exception(f"Sorry, {func} is not valid")

def activationDerivative(func: str, Z: np.ndarray) -> np.ndarray:
    """Return the derivative of the activation function given the function name and input"""
    if func == "sigmoid":
        return sigmoidDerivative(Z)
    elif func == "relu":
        return reluDerivative(Z)
    # elif func == "softmax":
    #     return softmax_derivative(Z)
    raise Exception(f"Sorry, {func} is not valid")


def lossFunction(func: str, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the loss given the loss function name, the activation, and the actual values"""
    if func == "mse":
        return np.mean(np.square(predicted - actual))
    raise Exception(f"Sorry, {func} is not valid")

def lossDerivative(func: str, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the derivative of the loss function given the loss function name, the activation, and the actual values"""
    if func == "mse":
        return 2 * (predicted - actual)
    raise Exception(f"Sorry, {func} is not valid")

def initParams(layer_dims: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Initializes the weights and biases for a neural network with random values.

    This function generates random weights and biases for each layer in the neural network 
    based on the specified layer dimensions. The weights are initialized with a normal 
    distribution and the biases are initialized similarly with random values.

    Args:
        layer_dims (list[int]): A list containing the number of nodes (neurons) in each layer, 
                                 including the input layer, hidden layers, and output layer.
                                 The length of this list represents the total number of layers.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists:
            - A list of weight matrices, where each element corresponds to the weights 
              for the connections between two consecutive layers.
            - A list of bias vectors, where each element corresponds to the biases for a given layer.

    Example:
        INPUT_LAYER_SIZE = 3
        OUTPUT_LAYER_SIZE = 2        
        weights, biases = initParams([INPUT_LAYER_SIZE, 7, 5, OUTPUT_LAYER_SIZE])
        # weights = [np.random.randn(4, 5), np.random.randn(3, 4), np.random.randn(2, 3)]
        # biases = [np.random.randn(4, 1), np.random.randn(3, 1), np.random.randn(2, 1)]
    
    Notes:
        - The function asserts that the input `layer_dims` list has at least two elements (input and output layers).
        - The function asserts that all elements in `layer_dims` are positive integers greater than 0

    """
    # Assert that layer_dims has at least two elements (input and output layers)
    assert len(layer_dims) > 1, "The layer_dims list must contain at least two elements (input and output layers)."
    
    # Assert that all elements in layer_dims are positive integers
    assert all(isinstance(dim, int) and dim > 0 for dim in layer_dims), "Each element in layer_dims must be a positive integer greater than zero."

    weights = []    # List of weight matrices
    biases = []     # List of bias vectors
    for i in range(0, len(layer_dims)-1):
        weights.append(np.random.randn(layer_dims[i+1], layer_dims[i]))
        biases.append(np.random.randn(layer_dims[i+1], 1))
    return weights, biases

def forwardPass(X: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray], hidden_func: str = "relu", output_func: str = "sigmoid", 
                    return_neurons_value:bool=False) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    """Perform the forward pass through the network.

    This function calculates the output of the neural network by propagating the input through 
    each layer, applying the appropriate activation function for each layer (hidden layers and output layer).


    Args:
        X (np.ndarray): The input to the network (shape: [input_size, 1]).
        weights (list[np.ndarray]): A list of weight matrices, where each element is the weight matrix for a given layer.
        biases (list[np.ndarray]): A list of bias vectors, where each element is the bias vector for a given layer.
        hidden_func (str, optional): The activation function for the hidden layers. Defaults to "relu".
        output_func (str, optional): The activation function for the output layers. Defaults to "sigmoid".
        return_neurons_value (bool, optional): Whether to return the activation values (neuron values) of each layer. Defaults to False.

    Returns:
        np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
            - If `return_neurons_value=False`, the output of the final layer is returned (shape: [output_size, 1]).
            - If `return_neurons_value=True`, a tuple is returned where:
              - The first element is the output of the final layer (shape: [output_size, 1]).
              - The second element is a list of neuron values (activations) for each layer (list of [layer_size, 1] arrays).

    Example:
        X = np.random.randn(5, 1)
        weights, biases = initParams([5, 3, 2])
        output, neurons = forwardPass(X, weights, biases, hidden_func="relu", output_func="sigmoid", return_neurons_value=True)
        print(f"Output: {output}")
        print(f"Neuron values: {neurons}")  # List of neuron activations for each layer

    Notes:
        - The function applies the activation function (`hidden_func`) to each hidden layer
        - The options for activation function are: "relu", "sigmoid"
    
    """
    n = len(biases) # Total number of layers
    neurons = []   # List of neuron values
    for i in range(n):
        if return_neurons_value:
            neurons.append(X)
        W = weights[i]
        B = biases[i]
        Z = np.dot(W, X) + B
        if n-1 == i:
            X = activationFunction(output_func, Z)
        else:
            X = activationFunction(hidden_func, Z)
    if return_neurons_value:
        return X, neurons
    return X


"""

    For BackPropogation:

    v1
            x1
    v2              h1
            x2              y
    v3              h2
            x3
    v4

    next_layer_delta = (Activation Derivative in to neuron) * (next_layer_delta * times the weight of the connection)
    The matrix would have the change required for each path, hence the sum of the matrix would be the change required for the neuron
    For weights, the change required would be the dot product of the next_layer_delta and the neuron value
    For biases, the change required would be the next_layer_delta

"""

#https://www.youtube.com/watch?v=kbGu60QBx2o
#https://www.youtube.com/watch?v=tIeHLnjs5U8&t=3s
def backProp(X: np.ndarray, Y:np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray],
                hidden_func: str = "relu", output_func: str = "sigmoid", loss_func: str = "mse")   -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Performs backpropagation to compute the gradients of the weights and biases for training a neural network.

    This function calculates the gradients of the loss function with respect to the weights and biases 
    using the chain rule of calculus. The gradients are later used to update the parameters during the training process by the `updateParams` function

    Args:
        X (np.ndarray): The input data (features) of shape (n_features, 1).
        Y (np.ndarray): The true labels (targets) of shape (n_output_units, 1).
        weights (list[np.ndarray]): A list of weight matrices, where each element is the weight matrix for a given layer.
                                    The weight at index 0 corresponds to the connection between the input and the first hidden layer.
        biases (list[np.ndarray]): A list of bias vectors, where each element corresponds to the biases for each layer.
        hidden_func (str, optional): The activation function used in the hidden layers. Options are:
                                     "relu" (default), "sigmoid", "tanh", etc.Default is "relu"
        output_func (str, optional): The activation function used in the output layer. Default is "sigmoid".
        loss_func (str, optional): The loss function used for computing the error. Default is "mse".

    Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists:
            - Gradients of the weights, where each element is the gradient of the weights for a given layer.
            - Gradients of the biases, where each element is the gradient of the biases for a given layer.

    Example:
            INPUT_LAYER_SIZE = 3
            OUTPUT_LAYER_SIZE = 2
            X = np.random.randn(INPUT_LAYER_SIZE, 1)
            weights, biases = initParams([INPUT_LAYER_SIZE, 7, 5, OUTPUT_LAYER_SIZE])
            Y = np.random.randn(OUTPUT_LAYER_SIZE, 1)
            delta_W, delta_B =  backProp(X, Y, weights, biases)
    
    Notes:
        - The options for activation function are: "relu", "sigmoid"
        - The options for loss function are: "mse"
        - This function takes one example at a time.
        - The gradients computed here are used to adjust the weights and biases during training by the `updateParams` function.
        - The activation function for all the hidden layers are same
    """
    predicted, neurons = forwardPass(X, weights, biases, hidden_func, output_func, return_neurons_value=True)
    n = len(weights)
    delat_W = []
    delat_B = []

    next_layer_delta = lossDerivative(loss_func, predicted, Y) * activationDerivative(output_func, predicted)
    
    delat_W.append(np.dot(next_layer_delta, neurons[-1].T))
    delat_B.append(next_layer_delta)

    for i in range(n-2, -1, -1):
        next_layer_delta = activationDerivative(hidden_func, neurons[i+1]) * np.dot(next_layer_delta.T, weights[i+1])
        next_layer_delta = np.sum(next_layer_delta, axis=0, keepdims=True).T
        delat_W.append(np.dot(next_layer_delta, neurons[i].T))
        delat_B.append(next_layer_delta)
    delat_W = delat_W[::-1]
    delat_B = delat_B[::-1]
    return delat_W, delat_B

def batchBackProp(X:list[np.ndarray], Y:list[np.ndarray], weights:list[np.ndarray], biases:list[np.ndarray],
                    hidden_func: str = "relu", output_func: str = "sigmoid", loss_func: str = "mse") -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Performs batch backpropagation to compute the average gradients of weights and biases for training a neural network.

    This function computes the gradients of the loss function with respect to the weights and biases by averaging 
    the gradients over a batch of input-output pairs. The gradients are accumulated and returned, which can be used 
    for parameter updates during training (e.g., gradient descent).

    Args:
        X (list[np.ndarray]): A list of input data arrays (features). Each element represents the input data for one training sample.
        Y (list[np.ndarray]): A list of true label arrays (targets). Each element represents the true labels for one training sample.
        weights (list[np.ndarray]): A list of weight matrices, where each element is the weight matrix for a given layer.
                                    The weight at index 0 corresponds to the connection between the input and the first hidden layer.
        biases (list[np.ndarray]): A list of bias vectors, where each element corresponds to the biases for each layer.
        hidden_func (str, optional): The activation function used in the hidden layers. Options are:
                                     "relu" (default), "sigmoid", "tanh", etc.Default is "relu"
        output_func (str, optional): The activation function used in the output layer. Default is "sigmoid".
        loss_func (str, optional): The loss function used for computing the error. Default is "mse".

    Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists:
            - Gradients of the weights, where each element is the gradient of the weights for a given layer.
            - Gradients of the biases, where each element is the gradient of the biases for a given layer.

    Example:
        INPUT_LAYER_SIZE = 3
        OUTPUT_LAYER_SIZE = 2        
        X_batch = [np.random.randn(INPUT_LAYER_SIZE, 1), np.random.randn(INPUT_LAYER_SIZE, 1)]
        Y_batch = [np.random.randn(OUTPUT_LAYER_SIZE, 1), np.random.randn(OUTPUT_LAYER_SIZE, 1)]
        weights, biases = initParams([INPUT_LAYER_SIZE, 7, 5, OUTPUT_LAYER_SIZE])
        delta_W, delta_B =  batchBackProp(X_batch, Y_batch, weights, biases)

    Notes:
        - This function assumes that each input-output pair in the batch corresponds to the same neural network structure.
        - The function iterates over all input-output pairs in the batch, computes the gradients for each pair using `backProp`,
            and accumulates these gradients before returning the averaged result.
        - The options for activation function are: "relu", "sigmoid"
        - The options for loss function are: "mse"
        - This function can take multiple examples at a time.
        - The gradients computed here are used to adjust the weights and biases during training by the `updateParams` function.
        - The activation function for all the hidden layers are same
    """
    delta_W = [np.zeros_like(w) for w in weights]
    delat_B = [np.zeros_like(b) for b in biases]
    for i in range(len(X)):
        del_w, del_b = backProp(X[i], Y[i], weights, biases, hidden_func, output_func, loss_func)
        for j in range(len(delta_W)):
            delta_W[j] += del_w[j]
            delat_B[j] += del_b[j]
    return delta_W, delat_B

def updateParams(weights: list[np.ndarray], biases: list[np.ndarray], 
                    delat_W: list[np.ndarray], delat_B: list[np.ndarray], alpha: float = 0.01) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """    Update the weights and biases of the neural network using the gradients.

    This function applies gradient descent to update the weights and biases 
    by subtracting the product of the learning rate (`alpha`) and the gradient of the loss function (`delta_W`, `delta_B`) 
    for each layer.

    Args:
        weights (list[np.ndarray]): A list of weight matrices, where each element corresponds to the weight matrix for a given layer.
        biases (list[np.ndarray]): A list of bias vectors, where each element corresponds to the bias vector for a given layer.
        delat_W (list[np.ndarray]): A list of gradients of the weights with respect to the loss function, computed using backpropagation.
        delat_B (list[np.ndarray]): A list of gradients of the biases with respect to the loss function, computed using backpropagation.
        alpha (float): The learning rate, which controls the step size during the update. Default is 0.01.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: The updated weights and biases after applying the gradient updates.

      Example 1: **Batch Update**
        delta_W, delta_B =  batchBackProp(X_batch, Y_batch, weights, biases)
        updateParams(weights, biases, delta_W, delta_B, alpha=0.01)

      Example 2: **Single Example Update**
        delta_W, delta_B =  backProp(X, Y, weights, biases)
        updateParams(weights, biases, delta_W, delta_B, alpha=0.01)


        
        delta_W, delta_B =  batchBackProp(X, Y, weights, biases)
        updateParams(weights, biases, delta_W, delta_B, alpha=0.01)


    Notes:
        - This function assumes that the dimensions of `delta_W` and `delta_B` match the dimensions of `weights` and `biases` respectively.
        - The function applies the gradient descent update rule: 
          `weights[i] -= alpha * delta_W[i]` and `biases[i] -= alpha * delta_B[i]` for each layer.
        - The update rule works the same for both single and batch updates, as long as `delta_W` and `delta_B` are appropriately computed.

    """
    for i in range(len(weights)):
        weights[i] -= alpha * delat_W[i]
        biases[i] -= alpha * delat_B[i]
    return weights, biases


def trainNetwork(X: list[np.ndarray], Y: list[np.ndarray], weights: list[np.ndarray], biases: list[np.ndarray],
                    hidden_func: str = "relu", output_func: str = "sigmoid", loss_func: str = "mse", 
                    alpha: float = 0.01, epochs: int = 1000, print_after_iterations:int = 100) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Train the neural network over a specified number of epochs using batch gradient descent.

    This function performs the training of the neural network by computing gradients 
    using backpropagation (via `batchBackProp`), updating the weights and biases 
    using gradient descent (via `updateParams`), and printing the loss every 100 epochs.

    Args:
        X (list[np.ndarray]): A list of input samples for the training. Each element of the list is a training example.
        Y (list[np.ndarray]): A list of target outputs corresponding to the inputs `X`.
        weights (list[np.ndarray]): A list of weight matrices for the network.
        biases (list[np.ndarray]): A list of bias vectors for the network.
        hidden_func (str, optional): The activation function for the hidden layers. Defaults to "relu".
        output_func (str, optional): The activation function for the output layer. Defaults to "sigmoid".
        loss_func (str, optional): The loss function used for computing the error. Defaults to "mse".
        alpha (float, optional): The learning rate for the gradient descent. Defaults to 0.01.
        epochs (int, optional): The number of training epochs. Defaults to 1000.
        print_after_iterations (int, optional): The frequency (in epochs) at which the loss should be printed. Does not print when set to 0. Defaults to 100

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing the updated weight matrices and bias vectors after training.

    Example:
        weights, biases = initParams([5, 4, 2])
        weights, biases = trainNetwork(X, Y, weights, biases, hidden_func="relu", output_func="sigmoid", loss_func="mse", alpha=0.01, epochs=1000, print_after=200)

    Notes:
        - This function performs batch gradient descent, meaning it updates the weights and biases 
          after processing all examples in the entire training set.
        - The training process involves computing the gradients using `batchBackProp`, updating the 
          weights and biases using `updateParams`, and printing the loss at every 100th epoch.
        - The loss is computed using the specified `loss_func`, which is passed to the `lossFunction` method.
        - If `print_after_iterations` is set to 0, the function does not print anything.
    """
    for i in range(epochs):
        delta_W, delta_B = batchBackProp(X, Y, weights, biases, hidden_func, output_func, loss_func)
        weights, biases = updateParams(weights, biases, delta_W, delta_B, alpha)
        if print_after_iterations and i % print_after_iterations == 0:
            print(f"Epoch: {i}, Loss: {lossFunction(loss_func, forwardPass(X[0], weights, biases), Y[0])}")
    return weights, biases

if __name__ == "__main__":
    INPUT_LAYER_SIZE = 5
    OUTPUT_LAYER_SIZE = 20
    X = np.random.randn(INPUT_LAYER_SIZE, 1)
    weights, biases = initParams([INPUT_LAYER_SIZE, 10, 5, OUTPUT_LAYER_SIZE])
    Y = np.random.randn(OUTPUT_LAYER_SIZE, 1)
    original =  forwardPass(X, weights, biases)
    # print("Original result", original)
    weights, biases = trainNetwork([X], [Y], weights, biases, print_after_iterations=0)
    final =  forwardPass(X, weights, biases)
    # print("Result after training", final)
    # print("Actual answer", Y)
    og_error = lossFunction("mse", original, Y)
    final_error = lossFunction("mse", final, Y)
    print("Improvement", (og_error - final_error)/og_error*100, "%")
