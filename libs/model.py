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

from libs.layers.grm import *
from libs.layers.embedder import *
from libs.layers.seed import *

import random
random.seed(1)
torch.manual_seed(1)


class LNSN(nn.Module):

    def __init__(self, n_spix, args):
        super().__init__()
        self.n_spix = n_spix
        self.sp_num = n_spix
        self.is_dilation = args.is_dilation
        self.device = args.device

        self.seed_strategy = args.seed_strategy

        
        self.train = True
        self.kn = args.kn
        ###########Optimizer Parameter##########

        self.embedder = Embedder(self.is_dilation)

        self.generater = SeedGenerater(self.n_spix, self.device, seed_strategy=self.seed_strategy)

        self.grm = GRM(args)


        #############init#######################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bd):
        

        b, _, h, w = x.size()
        ##########Feature Extracting###########
        f = self.embedder(x)

        #########Seed Generate#######
        cx, cy, probs = self.generater(f)


        if self.train:

            f, recons = self.grm(f, bd)

            f = f.view(b, -1, h*w)

            return recons, cx, cy, f, probs
        else:
            f = f.view(b, -1, h*w)
            return cx, cy, f, probs





