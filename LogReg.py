import numpy as np

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def Cost(X, y, beta):  # cost function
    scores = np.dot(X, beta)
    return np.sum(y*scores - np.log(1 + np.exp(scores)) )
    

def dCost(X, y, p): # derivative of cost function
    return - np.dot(X.T, y - p)


def test_prediction(X_test, y_test, weights): # tests results of classification against labels
    pred = sigmoid(np.dot(X_test, weights))
    pred = np.where(pred<0.5, 0 , 1)
    return np.mean(pred==y_test)

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
        if i == int(max_iter)-1:
            print (('tolerance not reached after {} iterations.'.format(i)),
                   'Cost gradient norm= {} '.format(np.linalg.norm(gradient)))
    return beta
        