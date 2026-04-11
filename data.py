import numpy as np
from train import train
from model import LogisticRegression
from utils import save_model, load_model

def load_txt(filepath,delimiter=None):
    data = np.loadtxt(filepath,delimiter=delimiter)
    X = data[:,:-1]
    y = data[:,-1].reshape(-1,1)
    return X,y












































