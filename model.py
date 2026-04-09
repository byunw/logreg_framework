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

    # pick up from here
    def predict(self,X,threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


# model = LogisticRegression(3)
# #test code for the model creation code
# assert np.all(model.weights==0)
# assert model.bias == 0
#
# #test code for predict_prob()
# X = np.array([[25,5000,3000],[40,8000,3000],[3,6,4000]])
# probabilities = model.predict_proba(X)
# assert probabilities[0,0] == 0.5
# assert probabilities[1,0] == 0.5
# assert probabilities[2,0] == 0.5
#
# classify_result = model.predict(X)
# assert classify_result[0,0] == 0
# assert classify_result[1,0] == 0
# assert classify_result[2,0] == 0
# print(classify_result)
#write once more test code for model.py
# model = LogisticRegression(3)
# assert np.all(model.weights==1)
# assert model.bias == 0
# X = np.array([[25,5000,3000],[40,8000,3000],[3,6,4000]])
# probabilites = model.predict_proba(X)
# assert probabilites[0,0] == 1
# classify_result = model.predict(X)
# assert classify_result[0,0] == 1






















