import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms

class MRData():
    """This class used to load MRnet dataset from `./images` dir
    """

    def __init__(self, plane, task = 'acl', train = True, transform = None, weights = None):
        """Initialize the dataset

        Args :
            plane : along which plane to load the data
            task : for which task to load the labels
            train : whether to load the train or val data
            transform : which transforms to apply

        """

        self.records = None
        if train:
            self.records = pd.read_csv('./images/train-{}.csv'.format(task))
            self.image_path = './images/train/{}/'.format(task)
        else:
            transform = None
            self.records = pd.read_csv('./images/valid-{}.csv'.format(task))
            self.image_path = './images/valid/{}/'.format(task)

        
        self.transform = transform 

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
            
        self.paths = [self.image_path +
                      '.npy' for filename in self.records['id'].tolist()]

        self.labels = self.records['label'].tolist()
        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.records)

    def __getitem__(self, index):
        """
        Returns `(image,label)` pair
        """
        img_raw = np.load(self.paths[index])
        label = self.labels[index]
        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])

        # apply transforms if possible, or else stack 3 images together
        # Note : if applying any transformation, use 3 to generate 3 images
        # but they should be almost similar to each other
        if self.transform:
            img_raw = self.transform(img_raw)
        else:
            img_raw = np.stack((img_raw,)*3, axis=1)
            img_raw = torch.FloatTensor(img_raw)

        return img_raw, label

    def pre_epoch_callback(self, epoch):
        """Callback to be called before every epoch.
        """
        pass

    def post_epoch_callback(self, epoch):
        """Callback to be called after every epoch.
        """
        pass