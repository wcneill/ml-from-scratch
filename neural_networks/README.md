# Deep Learning Directory Files:

1. __nn.py__ A from-scratch deep neural network class written entirely in Numpy. This class was written in an effort to understand the math behind neural networks from the ground up. 

2. __based_sequential.py__ A small extension of `torch.nn.Sequential` for ease of use in generating and training sequential models. Its `__init__` simply takes a tuple containing as elements the number of nodes at each desired layer. It then generates and registers ReLU modules for the inner layers and a log softmax activation for the final layer.  It contains a training method as well. 

Additionally this directory contains the following subdirectories:

1. __learning pytorch__: A collection of jupyter notebooks that iteratively work through the basics of PyTorch. The first notebook is a primer on the Tensor data structure, and each follow on notebook builds on the previous in order to build and train neural networks in the later notebooks. 