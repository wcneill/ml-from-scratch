import torch
from torch import nn, optim
from torchvision import datasets, transforms
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
        self.layers = layers
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

        for i in range(n_layers - 2):
            modules[f'fc{i}'] = nn.Linear(layers[i], layers[i + 1])
            modules[f'ReLu{i}'] = nn.ReLU()
            modules[f'drop{i}'] = nn.Dropout(p=0.2)
        modules['fc_out'] = nn.Linear(layers[-2], layers[-1])
        modules['smax_out'] = nn.LogSoftmax(dim=1)

        return modules

        return modules

    def train_model(self, trainload, epochs, criterion=nn.NLLLoss(), optimizer=optim.Adam, lr=0.003, do=.2, testload=None):
        """
        Train network parameters for given number of epochs.

        :param trainload: a DataLoader object containing training variables and targets used for training.
        :param testload: Optional. a DataLoader containing the validation set. If included, both training and
            validation loss will be tracked and can be plotted using model.plot_loss().
        :param epochs: Number of times the network will view the entire dataset
        :param optimizer: Learning method. Default optim.Adam
        :param lr: Learning Rate. Default 0.003
        :param do: Dropout rate. Default 0.2
        :param criterion: Loss function. defualt nn.NLLLoss,
        :param plot_loss: Set to True for a plot of training loss vs epochs.
        :return:
        """
        opt = optimizer(self.parameters(), lr)

        self.testload = testload
        self.train_loss = []
        self.valid_loss = []
        self.accuracy = []

        for e in range(epochs):
            running_tl = 0
            running_vl = 0
            running_ac = 0
            for x, y in trainload:
                x = x.view(x.shape[0], -1)
                opt.zero_grad()
                loss = criterion(self(x), y)
                loss.backward()
                opt.step()
                running_tl += loss.item()

            if testload is not None:
                self.eval()
                with torch.no_grad():
                    for x, y in testload:
                        x = x.view(x.shape[0], -1)
                        lps = self(x)
                        ps = torch.exp(lps)
                        loss = criterion(lps, y)
                        _, topclass = ps.topk(1, dim=1)
                        acc = (topclass == y.view(*topclass.shape)).numpy()
                        running_ac += acc.mean()
                        running_vl += loss.item()
                    self.accuracy.append(running_ac / len(testload))
                    self.valid_loss.append(running_vl / len(testload))

            self.train()
            self.train_loss.append(running_tl / len(trainload))

    def plot_loss(self):
        """
        Plot training loss per epoch. Will also plot validation loss if a validation set was included.
        """
        if self.valid_loss:
            plt.plot(self.valid_loss, label='Validation Loss')
        plt.plot(self.train_loss, label='Training Loss')
        plt.legend()
        plt.show()

    def save(self, filepath):
        checkpoint = {
            'layers': self.layers,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, filepath)


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


    layers = (784, 256, 128, 64, 10)
    model = LSequential(layers)
    model.train_model(trainloader, 2, testload=testloader)
    model.plot_loss()

    model.save('test_save.pth')
