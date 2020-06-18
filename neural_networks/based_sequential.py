from torch import nn, optim
from collections import OrderedDict
import matplotlib.pyplot as plt


class LSequential(nn.Sequential):
    """
    A Small extension for ease of use of PyTorch's nn.Sequential model. Rather than define each layer's
    activations one at a time, the model is passed a tuple on initialization. The tuple contains the number of
    nodes desired at each layer. The model is then automatically initialized with ReLU activations at each inner
    layer and a log softmax activation at the final layer.

    All other behavior should match that of `torch.nn.Sequential`
    """
    def __init__(self, layers):
        super().__init__(self.init_modules(layers))

    def init_modules(self, layers):
        """
        A helper method that creates sequential modules and adds them to an `OrderedDict` object for passing
        to the super class `nn.Sequential`
        :param layers: Tuple where each element is the number of nodes in the corresponding layer
        :return: OrderedDict containing activation modules for passing to `nn.sequential`
        """
        n_layers = len(layers)
        modules = OrderedDict()

        # Layer definitions for input and inner layers:
        for i in range(n_layers - 2):
            modules[f'fc{i}'] = nn.Linear(layers[i], layers[i + 1])
            modules[f'relu{i}'] = nn.ReLU()

        # Definition for output layer:
        modules['fc_out'] = nn.Linear(layers[-2], layers[-1])
        modules['smax_out'] = nn.LogSoftmax(dim=1)

        return modules

    def train(self, trainloader, epochs, optimizer=optim.Adam, lr=0.01, criterion=nn.NLLLoss(), plot_loss=False):
        """
        Train network parameters for given number of epochs.


        :param trainloader: a DataLoader object containing training variables and targets.
        :param epochs: Number of times the network will view the entire dataset
        :param optimizer: Learning method. Default optim.Adam
        :param lr: Learning Rate. Default 0.01
        :param criterion: Loss function. defualt nn.NLLLoss,
        :param plot_loss: Set to True for a plot of training loss vs epochs.
        :return:
        """

        opt = optimizer(self.parameters(), lr=lr)
        epoch_error = []
        for e in range(epochs):
            running_loss = 0
            for x, y in trainloader:
                x = x.view(x.shape[0], -1)
                opt.zero_grad()
                loss = criterion(self(x), y)
                loss.backward()
                opt.step()
                running_loss += loss.item()
            epoch_error.append(running_loss / len(trainloader))

        if plot_loss:
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.plot(epoch_error)