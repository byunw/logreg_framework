import numpy as np
from train import train
from model import LogisticRegression


def load_txt(filepath,delimiter=None):
    data = np.loadtxt(filepath,delimiter=delimiter)
    X = data[:,:-1]
    y = data[:,-1].reshape(-1,1)
    return X,y

#test code
X,y = load_txt("train.txt")
assert np.array_equal(X,np.array([[25,5000,3000],[40,8000,3000]]))
assert np.array_equal(y,np.array([[1],[1]]))

#full test code
X,y = load_txt("train.txt")
model1 = LogisticRegression(3)
print(model1.weights)
print(model1.bias)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
train(model1,X,y)
print(model1.weights)
print(model1.bias)







































