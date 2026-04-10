import numpy as np
from model import LogisticRegression

def save_model(model,filepath):
    np.savez(filepath,weights=model.weights,bias=model.bias)

def load_model(filepath):
    parameters = np.load(filepath)
    weights = parameters["weights"]
    #bias = float(parameters["bias"]
    bias = parameters["bias"]
    model = LogisticRegression(weights.shape[0])
    model.weights = weights
    model.bias = bias
    return model







