import numpy as np

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def Cost(X, y, beta):  # cost function
    scores = np.dot(X, beta)
    return np.sum(y*scores - np.log(1 + np.exp(scores)) )
    

def dCost(X, y, p): # derivative of cost function
    return - np.dot(X.T, y - p)

def LogisticRegression(X, y, l_rate, tol, max_iter): #logistic regresion with gradient descent
    beta = np.zeros(X.shape[1]) # prediction array
    counter = 0
    for i in range(int(max_iter)):
        p = sigmoid(np.dot(X, beta))
        gradient = dCost(X, y, p)
        beta -= l_rate * gradient
        counter += 1
        if counter%50000 == 0: # Check gradient every 50 000 steps
            if np.linalg.norm(gradient) < tol:
                print ('tolerance reached after {} iterations'.format(i))
                break
    return beta
        