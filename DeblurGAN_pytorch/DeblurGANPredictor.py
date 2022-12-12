import os
import argparse
import cv2
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import torch
from torchvision import transforms
import numpy as np
import model.model as module_arch
from utils.util import denormalize
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Deblurer:
    def __init__(self,resume="checkpoint-epoch300.pth"):
        # load checkpoint
        checkpoint = torch.load(resume)
        self.config = checkpoint['config']

        # build model architecture
        generator_class = getattr(module_arch, self.config['generator']['type'])
        self.generator = generator_class(**self.config['generator']['args'])

        # prepare model for deblurring
        self.generator.to(device)

        self.generator.load_state_dict(checkpoint['generator'])

        self.generator.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ])
        

    def deblur(self,img_path):
        # start to deblur
        blurred = Image.open(img_path).convert('RGB')
        # h,w 
        h = blurred.size[1]
        w = blurred.size[0]
        new_h = h - h % 4 + 4 if h % 4 != 0 else h
        new_w = w - w % 4 + 4 if w % 4 != 0 else w
        blurred = transforms.Resize([new_h, new_w], Image.BICUBIC)(blurred)
        if self.transform:
            blurred = self.transform(blurred)
        with torch.no_grad():

            blurred = blurred.to(device)
            # blurred = blurred.unsqueeze(0)
            deblurred = self.generator(blurred)
            deblurred_img = to_pil_image(denormalize(deblurred).squeeze().cpu())
            # from Image form to OpenCv form
            deblurred_img  = cv2.cvtColor(np.asarray(deblurred_img ),cv2.COLOR_RGB2BGR)
            return deblurred_img
