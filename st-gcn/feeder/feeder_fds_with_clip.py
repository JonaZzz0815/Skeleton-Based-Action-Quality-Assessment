# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import torch.utils.data as data

from PIL import Image
from feeder.transforms import trans, trans2

# visualization
import time

# operation
from . import tools

def get_clips(img_paths:list):
    images = list() 
    for img_path in img_paths:
        img = [Image.open(img_path).convert('RGB')]
        images.extend(img)
    
    process_data = trans(list(images))
    return process_data




class Feeder_fds_with_clip(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        rgb_path:the path to '.npy' data, the shape of data should be ??
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 diff_path,
                 rgb_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.diff_path = diff_path
        self.rgb_path = rgb_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        
        # load difficulty
        with open(self.diff_path, 'rb') as f:
            self.sample_name, self.diff = pickle.load(f)
        
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:100]
            self.diff = self.diff[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

        # load collection of clips of imgs
        self.rgb_col = np.load(self.rgb_path,allow_pickle=True)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        data_rgb = self.rgb_col[index]
        diff = self.diff[index]
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # ！！！！ here label is the score
        # load stacked img clip
        clip = get_clips(data_rgb)
        return data_numpy,float(label),float(diff),clip