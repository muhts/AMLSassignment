import numpy as np
import pandas as pd
import os
from skimage import io, transform
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

from classes.data_cleaner import removeNeg, replaceNeg, cleanFaces


def split_dataloader(name,label,batch=64):
    """
    Process csv to train, validate, test dataloaders

    Args:
        name - file name
        label - label
        batch - batch size
    """
    # reformats csv to dataframe
    df = readFormat(name)
    # gets rid of natural backgrounds
    df = cleanFaces(df)
    # only keeps file_name and label
    df = df[['file_name',label]]

    # getting classes
    # replaces -1 for binary tasks
    # removes -1 for multiclass tasks
    if label == 'smiling':
        classes = ['no_smile','smile']
        df = replaceNeg(df)
    elif label == 'eyeglasses':
        classes = ['no glassed','glasses']
        df = replaceNeg(df)
    elif label == 'young':
        classes = ['old','young']
        df = replaceNeg(df)
    elif label == 'human':
        classes = ['cartoon', 'human']
        df = replaceNeg(df)
    elif label == 'hair_color':
        classes = ['bald','blond','ginger','brown','black','grey']
        df = removeNeg(df)


    # splits dataset into train validate test (60,20,20)
    x_train_val, x_test, y_train_val, y_test = train_test_split(df['file_name'],
                                                                df[label],
                                                                test_size=0.20)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val,
                                                      y_train_val,
                                                      test_size=0.25)

    # converts dataset to torch Dataset object and applies transformations
    train_data = FacesDataset(x_train,y_train,transform = transforms.Compose([RandomCrop(224),
                                                                              transforms.ToTensor()
                                                                              ]))
    val_data = FacesDataset(x_val,y_val,transform = transforms.Compose([RandomCrop(224),
                                                                        transforms.ToTensor()
                                                                       ]))

    test_data = FacesDataset(x_test,y_test,transform=transforms.Compose([RandomCrop(224),
                                                                        transforms.ToTensor()
                                                                       ]))
    # send Dataset objects to dataloader
    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    return {'train': train_loader,
            'val': val_loader,
            'test' : test_loader,
            'classes' : classes}

def readFormat(name):
    """
    Read and correctly format csv

    Args:
        name - file name
    """
    df = pd.read_csv(name)
    df_reformat = {'file_name' :df['5000'][1:],
                   'hair_color':df['Unnamed: 1'][1:],
                   'eyeglasses':df['Unnamed: 2'][1:],
                   'smiling':df['Unnamed: 3'][1:],
                   'young':df['Unnamed: 4'][1:],
                   'human':df['Unnamed: 5'][1:],
                  }

    return pd.DataFrame(df_reformat,dtype=np.int)

def read_new(loc='testing_dataset'):
    loc = 'testing_dataset'
    files = glob.glob(loc +"/*.png")
    imgs = []
    file_name = []
    for myFile in files:
        im = io.imread(myFile)
        imgs.append(im)
        file_name.append(myFile[len(loc)+1:-4])
    return file_name

class FacesDataset(Dataset):
    """
    Loads dataset for use of dataloader in torch.utils

    Args:
        labels = dataframe containing file name [col 0] and label [col 1]
        root_dir = Location of images [default: 'dataset/']
        transform = transformations applied on dataset
    """

    def __init__(self,fileName,labels,root_dir="dataset/",transform=None):
        self.fileName = fileName
        self.labels = labels
        self.data_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        """
        Fetches image and applies any transforms
        Returns: transformed image and label
        """

        img_name = os.path.join(self.data_dir,
                                str(self.fileName.iloc[idx]))
        img = io.imread(img_name+".png")

        # applies transforms
        if self.transform:
            img = self.transform(img)

        return (img ,int(self.labels.iloc[idx]),self.fileName.iloc[idx])



class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):

        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(img, (new_h, new_w))

        return float(img)

class RandomCrop(object):
    """
    Randomly crop the image given

    Args:
        size : Desired output size.
               If int, does square crop.
    """

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]

        return img

class ToTensor(object):
    """
    Convert ndarrays images to Tensors
    """
    def __call__(self, img):
        # swap color axis
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)
        return img.float()
