import numpy as np

class Network:
    
    def __init__(self, 
                 dims, 
                 activation_function = 'ELU',
                 activation_function_param = 0.01, # used in the ELU activation function
                 task = 'classification'):
        '''
        Initialize network dimentions and methods.
        '''
        self.task = task
        self.dims = dims
        self.size = len(dims)
        self.weights = [np.random.randn(i, j)/np.sqrt(j) 
                        for i, j  in zip(dims[:-1], dims[1:])] # better weight initialization as per 
        self.bias = [np.random.randn(i) for i in dims[1:]]     # Nielsens book
        
        self.AF = ActivationFunction(activation_function,           #activation function 
                                     param = activation_function_param)
        self.CF = CostFunction(task)   # specify cost function for either classification or regression
        
    def feedforward(self, X):
        '''
        Feed a feature matrix through the network and set activations
        '''
        a = [0 for i in range(self.size)]
        z = [0 for i in range(self.size - 1)]
        a[0] = X
        for i in range(self.size-1):  #feedforward pass
            z[i] = np.matmul(a[i], self.weights[i]) + self.bias[i]
            a[i+1] = self.AF(z[i])
        self.z = z  # set activations
        self.a = a
    
    def backprop(self, X, y,l_rate, lmbd):
        '''
        calculate output errors and backpropogate
        to update weights and biases
        '''
        self.feedforward(X)
        z = self.z
        a = self.a
        delta = self.CF.output_error(z[-1], y) # calculate delta for apropriate cost function
        
        for i in range(self.size - 2, -1, -1): # backpropagate errors
            w_grad = np.matmul(a[i].T, delta) + lmbd*self.weights[i]
            b_grad = np.sum(delta, axis=0)
            delta = np.matmul(delta, self.weights[i].T)*self.AF(a[i], prime=True)
            self.weights[i] -= l_rate*w_grad # update weights
            self.bias[i] -= l_rate*b_grad
    
    def train(self,
              X_train,
              y_train,
              epochs = 200,
              batch_size = 50,
              l_rate = 0.001,
              lmbd = 0.7):
        '''
        Train network using batch gradient descent
        '''
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        inds = np.arange(n_samples)
        for epoch in range(epochs):
            for batch in range(n_batches):
                ind = np.random.choice(inds,
                                       size=batch_size, 
                                       replace=False)
                
                self.backprop(X_train[ind], y_train[ind],
                              l_rate, lmbd)
        
    def predict(self, X):
        '''
        feed a feature matrix (can be one or multiple samples) 
        and return network output
        '''
        self.feedforward(X)
        z_L = self.z[-1]
        
        if self.task == 'classification':
            return np.argmax(softmax(z_L), axis=1)
        
        elif self.task == 'regression':
            return sigmoid(z_L, 0)
    
    def accuracy(self, X, y):
        '''
        Calculate accuracy score for classification
        '''
        
        pred = self.predict(X)
        return np.sum(y == pred)/len(y)




class ActivationFunction(): 
    '''
    used to calculate activations in hidden layers, 
    the structure should make it relatively easy to 
    add more alternative activation functions
    '''
    
    def __init__(self, activation_function, param = 0.01):
        self.param = param
        strings = np.array(['sigmoid', 'ELU'])
        functions  = [sigmoid, ELU]
        ind = np.where(strings==activation_function)[0][0]
        self.function = functions[ind]
        
    def __call__(self, z, prime=False):
        return self.function(z, param=self.param, prime=prime)


'''
Supplementary functions for ActivationFunction class
'''        
def sigmoid(z, param, prime=False): # param is included but not used to be integrated
    sigm = np.exp(z)/(1 + np.exp(z))# into the ActivationFunction class
    if prime:
        return sigm*(1-sigm)
    else:
        return sigm
    
def ELU(z, param, prime=False):
    if prime:
        return np.where(z<0, param*np.exp(z), 1.)
    else:
        return np.where(z<0, param*(np.exp(z) - 1), z)
    

class CostFunction:
    '''
    used to calculate the activation in the last layer and
    the output error of the network.
    Not yet finished, as it only holds one activation funtion 
    at the moment, but could be extended with
    more alternatives.
    
    '''
    def __init__(self, cost_function):
        strings = np.array(['classification', 'regression'])
        functions  = [softmax_log_likelihood_delta, MSE_delta]
        ind = np.where(strings==cost_function)[0][0]
        self.function = functions[ind]
        
    def output_error(self, z, y):
        return self.function(z, y)
        
        
        


'''
Supplementary functions for CostFunction class
'''  
def softmax(z):
    z_exp = np.exp(z)
    return z_exp/np.sum(z_exp, axis=1, keepdims=True)

def softmax_log_likelihood_delta(z, y): # delta for classification
    return softmax(z) - y

def MSE_delta(z, y): # delta for regression
        return (sigmoid(z, 0) - y)*sigmoid(z, 0, prime=True)



        
def OneHot(input_vector):
    n_inputs = len(input_vector)
    n_categories = np.max(input_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), input_vector] = 1
    
    return onehot_vector
