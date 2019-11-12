import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import floor

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#function to visualize the images

def plot_image(index):
    '''plots the image when you give it an index and returns the correct output'''
    print(y_train[index]) # The label is 8
    plt.imshow(x_train[index], cmap='Greys')
'''
index = 59000
plot_image(index)
plt.show()'''

'''Need to stretch matrix of image into a vector of dim 784 (28x28)'''

def flatten(image):
    return image.flatten()

image = x_train[0]
print(flatten(image)[60:200])
print('\n')

def mini_batch(batch_size = 20,batch_number = 1,data_set = x_train):
    '''Creating mini_batches of a given size (user specified) so we can apply Stochastic Gradient Descent
        algorithm on it'''

    train_length = len(data_set)

    if batch_size>=train_length:
        print('maximum size of mini_batch can be: {}'.format(train_length))
    else:
        index = (batch_number - 1)*batch_size
        if index+batch_size-1>=train_length and train_length-1-index>0:
            return data_set[index:]
        elif train_length-1-index<=0:
            return 'batch_number is too large for the size for mini-batch provided. Adjust your batch_number to <= {}'.format(floor((train_length-1)/batch_size)+1)
        else:
            return data_set[index:index+batch_size]

# having a batch_size of 16-32 is always best
print(mini_batch(batch_size = 30,batch_number = 60)) #since there are only 60,000 training points

#----------------------------------------------------------------------------------------------
'''So my goal now is to shuffle the mnist training data x_train so that we can apply the NN
on all the mini_batches and then take the average over all these weights. Basically we will
be applying Stochastic Gradient Descent'''


batch_size = 5 #setting batch size to be small just to see everything is working
train_length = x_train.shape[0]

np.random.shuffle(x_train) #randomly shuffles the trainig data
#print(x_train)

no_of_batches = floor((train_length-1)/batch_size)+1
epochs = 10000


for number in range(1,no_of_batches+1):
    x_train_batch = mini_batch(batch_size,batch_number = number, data_set = x_train)
    y_batch = mini_batch(batch_size,batch_number = number, data_set = y_train)

def plot_image(index):
    'plots the image when you give it an index and returns the correct output'
    print(y_batch[index]) # The label is 8
    plt.imshow(x_train_batch[index], cmap='Greys')

for index in range(len(y_batch)):
    plot_image(index)
    plt.show()
