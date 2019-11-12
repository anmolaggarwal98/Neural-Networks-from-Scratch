
'''Different Activation functions'''
import numpy as np

class Tanh:
    def activation(self,z):
        return np.tanh(z)

    def derivative_activation(self,z):
        return (np.cosh(z))**(-2)

class Sigmoid:
    def activation(self,z):
        return 1/(1+np.exp(-z))

    def derivative_activation(self,z):
        return self.activation(z)*(1-self.activation(z))

class Relu:
    def activation(self,z):
        return np.maximum(0,z)

    def derivative_activation(self,z):
        z[z<=0 ] = 0
        z[z>0] = 1
        return z
