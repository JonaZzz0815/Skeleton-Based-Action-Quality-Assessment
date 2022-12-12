import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image

class FineDiving_Skeleton(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, dirname, train=True):
        super(FineDiving_Skeleton, self).__init__()
        if train:
            # 上面的读取数据函数load_cifar100传入使用
            self.labels, self.images = load_cifar100(f"{dirname}/train")
        else:
            self.labels,self.images = load_cifar100(f"{dirname}/test")
        self.images = self.images.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = image.astype(np.float32)
        image = torch.from_numpy(image).div(255.0)
        label = self.labels[index]
        label = int(label)
        return image, label