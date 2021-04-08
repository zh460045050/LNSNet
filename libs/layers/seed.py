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


class SeedGenerater(nn.Module):
    
    def __init__(self, n_spix, device, seed_strategy='grid'):
        super().__init__()
        
        self.c3_inorm_1 = nn.InstanceNorm2d(20, affine=True)
        self.c3_seeds_1 = nn.Conv2d(20, 20, 3, padding=1)
        self.c3_seeds_2 = nn.Conv2d(20, 3, 1)
        
        self.relu = nn.ReLU()
        
        
        self.sp_num = n_spix
        self.device = device
        self.seed_strategy = seed_strategy
    
    
    def seed_generate(self, spix):
        
        b, _, h, w = spix.size()
        
        S = h * w / self.sp_num
        sp_h = np.int32(np.floor(np.sqrt(S) / (w /  np.float(h))))
        sp_w = np.int32(np.floor(S / np.floor(sp_h)))
        
        
        
        spix = nn.AdaptiveAvgPool2d((np.int32(np.ceil(h / sp_h)), np.int32(np.ceil(w / sp_w))))(spix)
        spix = self.c3_seeds_1(spix)
        spix = self.c3_inorm_1(spix)
        spix = self.relu(spix)
        spix = self.c3_seeds_2(spix)
        
        prob = spix[:, 0].view(b, -1) #probability for seed
        prob = torch.sigmoid(prob)
        dx = spix[:, 1].view(b, -1) #x shift for seed
        dy = spix[:, 2].view(b, -1) #y shift for seed
        dx = torch.sigmoid(dx) - 0.5
        dy = torch.sigmoid(dy) - 0.5
        
        
        
        prob = prob.view(b, -1)
        ####Choosing the max prob in Grid as Seed#####
        sp_c = []
        
        for i in range(0, h, sp_h):
            for j in range(0, w, sp_w):
                start_x = i
                end_x = min(i + sp_h, h) - 1
                len_x = end_x - start_x + 1
                start_y = j
                end_y = min(j + sp_w, w) - 1
                len_y = end_y - start_y + 1
                
                x = (end_x + start_x) / 2.0
                y = (end_y + start_y) / 2.0
                
                ind = x*w + y
                sp_c.append(ind)
    
        sp_c = torch.from_numpy(np.array(sp_c)).long()
        
        o_cind = sp_c
        o_cx = torch.floor(o_cind / float(w))
        o_cy = torch.floor(o_cind - o_cx * w)
        if self.device == 'cuda':
            o_cx = o_cx.cuda()
            o_cy = o_cy.cuda()
        cx = torch.floor(o_cx + dx.view(-1) * sp_h * 2)
        cy = torch.floor(o_cy + dy.view(-1) * sp_w * 2)



        #print(dx[:, sp_c].view(-1))
        #print(dy[:, sp_c].view(-1))

        cx = cx.clamp(0, h-1)
        cy = cy.clamp(0, w-1)
            
            
        return cx, cy, prob
    
    def grid_seed(self, spix):
        
        b, _, h, w = spix.size()
        
        S = h * w / self.sp_num
        sp_h = np.int32(np.floor(np.sqrt(S) / (w /  np.float(h))))
        sp_w = np.int32(np.floor(S / np.floor(sp_h)))
        
        ####Choosing the max prob in Grid as Seed#####
        sp_c = []
        for i in range(0, h, sp_h):
            for j in range(0, w, sp_w):
                start_x = i
                end_x = min(i + sp_h, h) - 1
                len_x = end_x - start_x + 1
                start_y = j
                end_y = min(j + sp_w, w) - 1
                len_y = end_y - start_y + 1
                
                x = (end_x + start_x) / 2.0
                y = (end_y + start_y) / 2.0
                
                
                ind = x*w + y
                sp_c.append(ind)
    
        sp_c = torch.from_numpy(np.array(sp_c)).long()

        o_cind = sp_c
        o_cx = torch.floor(o_cind / float(w))
        o_cy = torch.floor(o_cind - o_cx * w)
        if self.device == 'cuda':
            o_cx = o_cx.cuda()
            o_cy = o_cy.cuda()

        cx = o_cx.clamp(0, h-1)
        cy = o_cy.clamp(0, w-1)


        return cx, cy, torch.ones(b, h*w)

    def forward(self, x):
        
        if self.seed_strategy == 'network':
            #seed_dis = self.c3_seeds(x)
            cx, cy, probs = self.seed_generate(x)
        elif self.seed_strategy == 'grid':
            cx, cy, probs = self.grid_seed(x)
        
        return cx, cy, probs
