import numpy as np

class LogisticRegression:
    def __init__(self,num_features):
        # self.weights = np.zeros((num_features,1))
        self.weights = np.zeros((num_features,1))
        self.bias = 0.0
    def sigmoid(self,z):
        return 1 / (1+np.exp(-z))
    def predict_proba(self,X):
        #Xw + bias
        z = np.dot(X,self.weights) + self.bias
        return self.sigmoid(z)
    def predict(self,X,threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)























