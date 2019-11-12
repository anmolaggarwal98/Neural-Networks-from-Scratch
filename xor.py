import DeepLearning
from DeepLearning import NN

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from math import floor
import pprint
import random
from activation import Tanh,Relu,Sigmoid

t = time.process_time()

X = np.array([[0,0,1,1],[0,1,0,1]])
Y = np.array([[0,1,1,0]])

blah = NN(X,Y)
epochs = 2000
parameters, A2, Cost = blah.backward_propagation(epochs = epochs)

elapsed_time = time.process_time() - t

pprint.pprint(parameters)
print('\nActivation Function: {}'.format(blah.__class__.__bases__[0].__name__))
print('\nDesired Output:',*blah.Y)
print("\nOutput from neural network after {} epochs: \
".format(epochs),end='')
print(*A2)
print('')
print('Code took {} secs to run'.format(round(elapsed_time,4)))
plt.plot(Cost);
plt.title('Cost Function after {} epochs: '.format(epochs))
plt.show()
plt.plot(X,'.', markersize=18);
plt.show()
