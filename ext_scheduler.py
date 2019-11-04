# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:29:31 2019

@author: Administrator
"""

import torch.nn as nn

def bn_momentum_setter_default(bn_momentum):
    
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
            
    return fn

class BNMomentum(object):
    
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=bn_momentum_setter_default):
        
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None, **kwargs):
        
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
