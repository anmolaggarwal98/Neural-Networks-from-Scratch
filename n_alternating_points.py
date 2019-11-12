import DeepLearning
from DeepLearning import NN

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import pprint
import random
from activation import Tanh,Relu,Sigmoid

t = time.process_time()

n = 3
X = np.array([[i] for i in np.linspace(0,1-2**(-n),2**n+1)])

Y=np.zeros((2**n+1,1))   #create y with 010101010
for i in range(int(2**n/2)):
    Y[2*i+1] = 1

blah1 = NN(X,Y,learning_rate = 0.1)
epochs = 500
parameters, A2, Cost = blah1.backward_propagation(epochs = epochs)

elapsed_time = time.process_time() - t

pprint.pprint(parameters)
print('\nActivation Function: {}'.format(blah1.__class__.__bases__[0].__name__))
print('\nDesired Output:',*blah1.Y)
print("\nOutput from neural network after {} epochs:\n ".format(epochs),end='')
print(*A2)
print('')
print('Code took {} secs to run'.format(round(elapsed_time,4)))
plt.plot(Cost);
plt.title('Cost Function after {} epochs: '.format(epochs))
plt.show()

plt.plot(X, A2);
plt.title('Plot sawtooth function with n = {} using {}\
 activation'.format(n,blah1.__class__.__bases__[0].__name__));
plt.show()
