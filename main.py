from utils import initParams, forwardPass, lossFunction, trainNetwork
import numpy as np

INPUT_LAYER_SIZE = 5
OUTPUT_LAYER_SIZE = 20
X = np.random.randn(INPUT_LAYER_SIZE, 1)
weights, biases = initParams([INPUT_LAYER_SIZE, 10, 5, OUTPUT_LAYER_SIZE])
Y = np.random.randn(OUTPUT_LAYER_SIZE, 1)
original =  forwardPass(X, weights, biases)
# print("Original result", original)
weights, biases = trainNetwork([X], [Y], weights, biases)
final =  forwardPass(X, weights, biases)
# print("Result after training", final)
# print("Actual answer", Y)
og_error = lossFunction("mse", original, Y)
final_error = lossFunction("mse", final, Y)
print("Improvement", (og_error - final_error)/og_error*100, "%")
