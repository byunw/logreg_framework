import numpy as np
from model import LogisticRegression

def compute_loss(y,p):

    #come back to these next 2 lines of code
    # epsilon = 1e-15
    # p = np.clip(p, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss

def train(model,X,y,lr=0.01,epochs=1,verbose=True):
    n_samples = X.shape[0]
    loss_history = []

    for epoch in range(epochs):
        p = model.predict_proba(X)
        loss = compute_loss(y,p)
        loss_history.append(loss)
        dw = (X.T @ (p-y)) / n_samples
        # print(dw)
        db = np.mean(p-y)
        # print(db)

        #weights/bias update
        model.weights -= lr * dw
        model.bias -= lr * db

        # print(model.weights)
        # print(model.bias)
    # print(loss_history)


#test code1
#training data samples
# X = np.array([[25,5000,3000],[40,8000,3000]])
# #truth labels for above training datasamples
# y = np.array([
#     [1],
#     [1],
# ]
# )
# model1 = LogisticRegression(3)
# p = model1.predict_proba(X)
# assert p[0,0] == 0.5
# assert p[1,0] == 0.5
# binary_cross_entropy_loss = compute_loss(y,p)
# print(binary_cross_entropy_loss)
# assert binary_cross_entropy_loss > 0.6
# assert binary_cross_entropy_loss < 0.7

#test code2
# X = np.array([[25,5000,3000],[40,8000,3000]])
# y = np.array([
#     [1],
#     [1],
# ])
#
# model2 = LogisticRegression(3)
# p = model2.predict_proba(X)
# print(p)
# train(model2,X,y)
















