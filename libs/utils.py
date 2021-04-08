from __future__ import absolute_import
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

import sys
import errno
import shutil
import json
import os.path as osp
from skimage import color
from skimage.segmentation._slic import _enforce_label_connectivity_cython

import random
random.seed(1)
torch.manual_seed(1)


def preprocess(image, device="cuda"):
    #image = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    #h, w = image.shape[-2:]
    #coord = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))).float()[None]

    #coord[:, 0, : , :] = coord[:, 0, : , :] / np.float(h)
    #coord[:, 1, : , :] = coord[:, 1, : , :] / np.float(w)
    #image = image / 255.0


    #input = torch.cat([image, coord], 1).to(device)
    #input = (input - input.mean((2, 3), keepdim=True)) / input.std((2, 3), keepdim=True)

    image = color.rgb2lab(image)

    image[:, :, 0] = image[:, :, 0] / np.float(128.0)
    image[:, :, 1] = image[:, :, 1] / np.float(256.0)
    image[:, :, 2] = image[:, :, 2] / np.float(256.0)
    

    image = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    h, w = image.shape[-2:]
    #print(h, w)
    if h > w:
        image = image.permute(0, 1, 3, 2)
    h, w = image.shape[-2:]
    #print(h, w)
    coord = torch.stack(torch.meshgrid(torch.arange( h ), torch.arange(w))).float()[None]
    #coord = coord / img.shape[-2:
    coord[:, 0, : , :] = coord[:, 0, : , :] / np.float(h) - 0.5
    coord[:, 1, : , :] = coord[:, 1, : , :] / np.float(w) - 0.5
    #print(coord)
    #print(image.shape)
    input = torch.cat([image, coord], 1).to(device)
    input = (input - input.mean((2, 3), keepdim=True)) / input.std((2, 3), keepdim=True)

    return input




def drawCenter(image, cxs, cys):

    for cx, cy in zip(cxs, cys):
        cv2.circle(image, (cy, cx), 2, (0, 0, 1.0), 4)
    return image
    


def read_list(listPath):
    images = []
    with open(listPath, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
                pass
            path = lines[:-1]
            images.append(path)

    return images


class Logger(object):
    """
        Write console output to external text file.
        Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
        """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()
        
    def __enter__(self):
        pass
        
    def __exit__(self, *args):
        self.close()
        
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
        
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

