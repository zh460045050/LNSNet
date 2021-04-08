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


class GALFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, Lambda, w):
        ctx.save_for_backward(input_, Lambda, w)
        output = input_
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        
        inputs, Lambda, w = ctx.saved_tensors# pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            
            #print(grad_output.shape)
            w_cp = w.squeeze()
            F_in, F_out = w_cp.size()
            w_c = torch.abs(w_cp[:3, :])
            w_s = torch.abs(w_cp[3:, :])
            w_c = torch.mean(w_c, 0)
            w_s = torch.mean(w_s, 0)
            #red = dw
            dw = w_s * w_c
            red = dw / (Lambda + dw)
            #print(1 - red)
            Lambda = 0.5 * dw + 0.5 * Lambda
            
            grad_input = grad_output * (1 - red.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
        return grad_input, None, None


class GBLFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, bd_map):
        ctx.save_for_backward(input_, bd_map)
        output = input_
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        
        inputs, bd_map = ctx.saved_tensors# pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            bd_map = 1 - bd_map
            #rate = 2 / (1 + np.exp(-10 * bd_map)) - 1
            lamda =  - 1 * bd_map#rate #* torch.mean(bd_map, [0,1])
            lamda[bd_map < 0.7] = 1
            #lamda[bd_map >= 0.5] = -rate[bd_map >= 0.5]
            #print(torch.mean(bd_map, [0,1]))
            grad_input = grad_output
            grad_input[:, 3, :, :] = grad_input[:, 3, :, :] * lamda
            grad_input[:, 4, :, :] = grad_input[:, 4, :, :] * lamda
        return grad_input, None


class GBLayer(nn.Module):
    def __init__(self):
        """
            A gradient reversal layer.
            This layer has no parameters, and simply reverses the gradient
            in the backward pass.
            """
        
        super().__init__()
    
    def forward(self, f, bd):
        
        return GBLFunction.apply(f, bd)

class GALayer(nn.Module):
    def __init__(self):
        """
            A gradient reversal layer.
            This layer has no parameters, and simply reverses the gradient
            in the backward pass.
            """
        
        super().__init__()
    
    def forward(self, f, Lambda, w):
        
        return GALFunction.apply(f, Lambda, w)


class GRM(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.use_gal = args.use_gal
        self.use_gbl = args.use_gbl
        if args.use_gal:
            self.Lambda = torch.autograd.Variable(torch.zeros(20 * 1 * 1).type(torch.float32), volatile=True)
        
        self.recons = nn.Conv2d(20, 5, 1)
        
        if args.use_gbl:
            self.gbl = GBLayer()
                
        if args.use_gal:
            self.gal = GALayer()

    
    def forward(self, f, bd):
        
        recons = self.recons(f)
        
        if self.use_gbl:
            recons = self.gbl(recons, bd)
        
        if self.use_gal:
            f = self.gal(f, self.Lambda, self.recons.weight)
        
        return f, recons






