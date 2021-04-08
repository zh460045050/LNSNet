import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as io
import cv2
import time

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from skimage.segmentation._slic import _enforce_label_connectivity_cython

from libs.model import *
from libs.utils import *

import random
random.seed(1)
torch.manual_seed(1)

def assignment_test(f, input, cx, cy, alpha=1):
    
    b, _, h, w = input.size()
    p = input[:, 3:, :, :]

    p = p.view(b, 2, -1)
    cind = cx * w + cy
    cind = cind.long()
    c_p = p[:, :, cind]
    c_f = f[:, :, cind]
    
    _, c, k = c_f.size()
    
    N = h*w
    
    dis = torch.zeros(b, k, N)
    for i in range(0, k):
        cur_c_f = c_f[:, :, i].unsqueeze(-1).expand(b, c, N)
        cur_p_ij = cur_c_f - f
        cur_p_ij = torch.pow(cur_p_ij, 2)
        cur_p_ij = torch.sum(cur_p_ij, dim=1)
        dis[:, i, :] = cur_p_ij
    dis = dis / alpha
    dis = torch.pow((1 + dis), -(alpha + 1) / 2)
    dis = dis.view(b, k, N).permute(0, 2, 1).contiguous() #b,N,k
    dis = dis  / torch.sum(dis, dim=2).unsqueeze(-1)

    return dis
