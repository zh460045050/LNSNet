import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as io
import cv2

from tensorboardX import SummaryWriter
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from skimage.segmentation._slic import _enforce_label_connectivity_cython

from libs.model import *
from libs.test import *

import time

import random
random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser()

######Data Setting#######
parser.add_argument('--img_path', default='demo_imgs/012.jpg', help='The path of the source image')
######Model Setting#######
parser.add_argument('--device', default='cpu', help='use cuda or cpu')
parser.add_argument('--n_spix', type=int, default=100, help='The number of superpixel')
parser.add_argument('--kn', type=int, default=16, help='The number of dis limit')
parser.add_argument('--seed_strategy', type=str, default='network', help='network/grid')
#####Optimizing Setting####
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--use_gal', default=True, help='Using sowm weight update')
parser.add_argument('--use_gbl', default=True, help='Using sowm weight update')
parser.add_argument('--is_dilation', default=True, help='Using dilation convolution')
parser.add_argument('--check_path', type=str, default='./lnsn_BSDS_checkpoint.pth.pth')

################

args = parser.parse_args()


model = LNSN(args.n_spix, args)

model.load_state_dict(torch.load(args.check_path))

img = plt.imread(args.img_path)
input = preprocess(img, args.device)


with torch.no_grad():
    
    b, _, h, w = input.size()
    recons, cx, cy, f, probs = model.forward(input, torch.zeros(h, w))
    spix = assignment_test(f, input, cx, cy) 

    spix = spix.permute(0, 2, 1).contiguous().view(b, -1, h, w)
    spix = spix.argmax(1).squeeze().to("cpu").detach().numpy()


segment_size = spix.size / args.n_spix
min_size = int(0.06 * segment_size)
max_size = int(3.0 * segment_size)
spix = _enforce_label_connectivity_cython(spix[None], min_size, max_size)[0]

if img.shape[:2] != spix.shape[-2:]:
    spix = spix.transpose(1, 0)

write_img = mark_boundaries(img, spix, color=(1, 0, 0))

plt.imsave("result_" + args.img_path.split('/')[-1], write_img)





