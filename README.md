# Neural-Networks-from-Scratch

Idea of this project was to understand the underlying mathematics behind a simple deep net (1 hidden layer) NN and then code this up in python without the use of any specialised libraries (like tensorflow) and then experiment with the algorithm on different problems/datasets for example:
* XOR Problem in 2D
* n-alternating points problem (approximating the saw-tooth function)
* MNIST datasets (using my NN to predict the letters of handwritten numbers from 0-9)

I have also tried to show the convergence rate depending on which activation function I use (I have provided 3 - tanh, sigmoid and relu in `activation.py` but more can be added by changing this file) which can be changed by changing the inheritance class in `DeepLearning.py`. I have completed the examples of `n_alternating_points.py` and `xor.py` but I am still working on `mnist.py` which will be completed in due course. With `mnist.py`, I have so far imported the mnist datasets, split it into mini-batches and now my goals is to write a code which runs NN on each mini-batch and then takes the average over all the weights I modify in each batch. 

The resources I used to understand the maths behind this are: `https://mlfromscratch.com/neural-networks-explained/#cost-function`
and `http://neuralnetworksanddeeplearning.com/chap2.html` and also my lectures notes found here: `https://courses.maths.ox.ac.uk/node/37111/materials`

I have even added a jupyter notebook showing all my code and graphs etc. Note I only imported the `tensorflow` library in order to import mnist datasets NOT to run code with it. 
