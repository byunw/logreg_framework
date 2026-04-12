import numpy as np

def compute_loss(y,p):
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
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
        db = np.mean(p-y)

        #weights/bias update
        model.weights -= lr * dw
        model.bias -= lr * db
















