# data-science-at-home
This repository contains a combinations of modules that fall into two categories:
- convenience methods that extend or automate common machine learning tasks. One example of this is LSequential.py found in the neural networks fold.
- from scratch implementations of common machine learning models. These are designed to reinforce understanding of the underlying math and statistics behind common machine learning algorithms found in high level libraries like Sklearn and PyTorch. For example, there is a from scratch decision tree algorithm, implemented using AnyTree and NumPy.

The repository is roughly organized as follows:

1. __linear_algebra__ - From scratch PCA module and other basic linear algebra practice. 
2. __decision_trees__ - A from scratch decision tree classifier for regression. 
3. __neural_networks__ - A collection of modules and notebooks created for extending PyTorch functionality. 
4. __data_sets__ - Custom extensions of PyTorch's `torch.utils.data.Datasets`, i.e for when a Kaggle dataset is formatted differently from PyTorch's own version of the same dataset.
