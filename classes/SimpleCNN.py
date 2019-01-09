import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

class EmotionCNN(nn.Module):
    """
    PyTorch simple CNN model with configured hyper parameters
    Model:
        1 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        2 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        3 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)

        Flatten

        1 - Hidden Layer (units = 500, Activation = ReLu)
        2 - output Layer (units = 2)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolutional layer (sees batchsize*3*224*224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # dense layers (fully connencted)
        self.fc1 = nn.Linear(64*28*28, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        x = x.view(-1, 64*28*28)
        x = self.dropout(x)

        # hidden layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # output layer
        x = self.fc3(x)
        return x

class AgeCNN(nn.Module):
    """
    PyTorch simple CNN model with configured hyper parameters
    Model:
        1 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        2 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        3 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)

        Flatten

        1 - Hidden Layer (units = 300, Activation = ReLu)
        2 - output Layer (units = 2)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolutional layer (sees batchsize*3*224*224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # dense layers (fully connencted)
        self.fc1 = nn.Linear(64*28*28, 300)
        self.fc2 = nn.Linear(300, 2)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        x = x.view(-1, 64*28*28)
        x = self.dropout(x)

        # hidden layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        return x

class GlassesCNN(nn.Module):
    """
    PyTorch simple CNN model with configured hyper parameters
    Model:
        1 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        2 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        3 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)

        Flatten

        1 - Hidden Layer (units = 500, Activation = ReLu)
        2 - output Layer (units = 2)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolutional layer (sees batchsize*3*224*224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # dense layers (fully connencted)
        self.fc1 = nn.Linear(64*28*28, 500)
        self.fc2 = nn.Linear(500, 2)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        x = x.view(-1, 64*28*28)
        x = self.dropout(x)

        # hidden layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        return x

class HumanCNN(nn.Module):
    """
    PyTorch simple CNN model with configured hyper parameters
    Model:
        1 - Conv layer (kernel =(3*3), same padding, Activation = ReLu)
        2 - Conv layer (kernel =(3*3), same padding, Activation = ReLu)
        3 - Conv layer (kernel =(3*3), same padding, Activation = ReLu)

        Flatten

        1 - Hidden Layer (units = 500, Activation = ReLu)
        2 - output Layer (units = 2)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolutional layer (sees batchsize*3*224*224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # dense layers (fully connencted)
        self.fc1 = nn.Linear(64*28*28, 500)
        self.fc2 = nn.Linear(500, 2)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        x = x.view(-1, 64*28*28)
        x = self.dropout(x)

        # hidden layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        return x

class HairColorCNN(nn.Module):
    """
    PyTorch simple CNN model with configured hyper parameters
    Model:
        1 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        2 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)
        3 - Conv layer (kernel =(5*5), same padding, Activation = ReLu)

        Flatten

        1 - Hidden Layer (units = 400, Activation = ReLu)
        2 - output Layer (units = 6)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolutional layer (sees batchsize*3*224*224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # dense layers (fully connencted)
        self.fc1 = nn.Linear(64*28*28, 500)
        self.fc2 = nn.Linear(500, 6)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        x = x.view(-1, 64*28*28)
        x = self.dropout(x)

        # hidden layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        return x
