import numpy as np
from train import train
from model import LogisticRegression
from utils import save_model, load_model

def load_txt(filepath,delimiter=None):
    data = np.loadtxt(filepath,delimiter=delimiter)
    X = data[:,:-1]
    y = data[:,-1].reshape(-1,1)
    return X,y

#test code
# X,y = load_txt("train.txt")
# assert np.array_equal(X,np.array([[25,5000,3000],[40,8000,3000]]))
# assert np.array_equal(y,np.array([[1],[1]]))
#full test code
# X,y = load_txt("train.txt")
# model1 = LogisticRegression(3)
# train(model1,X,y)
#
# save_model(model1,"model1")
# retrieved_model = load_model("model1.npz")
# print(retrieved_model.weights)
# print(retrieved_model.bias)
# assert retrieved_model.weights[0][0] == 0.1625
# assert retrieved_model.weights[1][0] == 32.5
# assert retrieved_model.weights[2][0] == 15
# assert retrieved_model.bias == 0.005












































