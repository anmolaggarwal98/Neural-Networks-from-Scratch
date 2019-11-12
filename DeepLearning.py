import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import pprint
import random
from activation import Tanh,Relu,Sigmoid

class NN(Tanh):
    def __init__(self,X,Y, learning_rate = 0.1):
        self.X = X
        self.Y = Y
        self.lr = learning_rate

    def initialize_parameters(self):
        '''Initializes all the parameters in NN and stores it in a dictionary'''

        X,Y = self.X, self.Y
        n_row = X.shape[0]   #gives me 2
        n_col = X.shape[1] #give me 4
        n_y = Y.shape[0]   #gives you 1, i.e. dimension of the output layer

        W1 = np.random.randn(n_col, n_row)
        b1 = np.zeros((n_col, 1))

        W2 = np.random.randn(n_y, n_col)
        b2 = np.random.randn(n_y, Y.shape[1])  #A1.shape[1] = 4

        parameters = {"W1" : W1, "b1": b1,
                      "W2" : W2, "b2": b2}
        return parameters

    def forward_propagation(self,parameters):
        X = self.X

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        Z1 = W1@X + b1
        A1 = self.activation(Z1)

        Z2 = W2@A1 + b2
        A2 = self.activation(Z2)
        #print(A2)

        cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
        return cache

    def backward_propagation(self,epochs = 10000):
        X, Y = self.X, self.Y
        parameters = self.initialize_parameters() #only needed at the start

        m = X.shape[1]
        Cost = np.zeros(epochs)

        for i in range(epochs):
            (Z1, A1, W1, b1, Z2, A2, W2, b2) = self.forward_propagation(parameters)
            delta2 = np.multiply((A2-Y),self.derivative_activation(Z2))

            Cost[i] = 0.5*np.sum((A2-Y)**2)   #cost function in each loop

            dW2 = delta2@A1
            db2 = delta2 #np.sum(dZ2, axis = 1, keepdims=True)

            delta1 = np.multiply(W2.T@delta2, self.derivative_activation(Z1))
            dW1 = X@delta1
            db1 = delta1

            #adjusting the weights and biases (i.e. nudging them slightly)
            W2 = W2 - delta2.dot(A1)*(self.lr/m)

            b2 = b2 - np.sum(db2, axis = 1, keepdims=True)*(self.lr/m)

            W1 = W1 - delta1.dot(X.T)*(self.lr/m)
            b1 = b1 - np.sum(db1, axis = 1, keepdims=True)*(self.lr/m)

            # the nudged weights and biases become your new parameters
            parameters = {"W1" : W1, "b1": b1,
                          "W2" : W2, "b2": b2}

        return parameters, A2, Cost
