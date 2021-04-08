import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as io
import cv2

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from skimage.segmentation._slic import _enforce_label_connectivity_cython

from libs.utils import *

import random
random.seed(1)
torch.manual_seed(1)


class Embedder(nn.Module):
    
    def __init__(self, is_dilation=True):
        super().__init__()
        
        self.is_dilation = is_dilation
        
        
        self.rpad_1 = nn.ReflectionPad2d(1)
        if self.is_dilation:
            self.c1_1 = nn.Conv2d(5, 10, 3, padding=0)
            self.c1_2 = nn.Conv2d(5, 10, 3, padding=0, dilation=1)
            self.c1_3 = nn.Conv2d(5, 10, 3, padding=1, dilation=2)
            self.c1_4 = nn.Sequential(nn.InstanceNorm2d(35, affine=True), nn.ReLU(), nn.ReflectionPad2d(1), nn.Conv2d(35, 10, 3, padding=0))
        else:
            self.c1 = nn.Conv2d(5, 10, 3, padding=0)
        self.inorm_1 = nn.InstanceNorm2d(10, affine=True)
        #self.inorm_1 = nn.BatchNorm2d(10, affine=True)
        
        self.rpad_2 = nn.ReflectionPad2d(1)
        self.c2 = nn.Conv2d(15, 20, 3, padding=0)
        #self.inorm_2 = nn.BatchNorm2d(20, affine=True)
        self.inorm_2 = nn.InstanceNorm2d(20, affine=True)
        
        self.relu = nn.ReLU()


    def forward(self, x):
        
        spix = self.rpad_1(x)
        if self.is_dilation:
            spix_1 = self.c1_1(spix)
            spix_2 = self.c1_2(spix)
            spix_3 = self.c1_3(spix)
            spix = torch.cat([x, spix_1, spix_2, spix_3], dim=1)
            spix = self.c1_4(spix)
        else:
            spix = self.c1(spix)
        
        spix = self.inorm_1(spix)
        spix = self.relu(spix)
        
        spix = torch.cat((spix, x), dim=1)
        
        spix = self.rpad_2(spix)
        spix = self.c2(spix)
        spix = self.inorm_2(spix)
        spix = self.relu(spix)
            
        return spix







