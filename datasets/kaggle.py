import torch
from torch.utils.data import Dataset
import pandas as pd


class MNIST(Dataset):
    """ Kaggle's MNIST dataset. Kaggle formats each image as a single row vector of pixels. This custom extension of
     the PyTorch `Dataset` class will allow the data to be split into training and validation sets by
      `torch.utils.random_split` and loaded into a PyTorch DataLoader."""
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to CSV file with image pixel data
            transform (callable, optional): Optional transform applied to the sample
        """
        self.pixel_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.pixel_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self.pixel_frame.iloc[index, 1:].to_numpy(dtype='float64').reshape(1, -1)
        label = self.pixel_frame.iloc[index, 0]
        if self.transform:
            image = self.transform(image)

        return image, label
