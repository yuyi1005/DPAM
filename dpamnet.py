# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:40:03 2019

@author: Administrator
"""

import torch
import torch.nn as nn
from collections import OrderedDict

class Dpam(nn.Module):
    
    def __init__(self, planes_tab=[], learnable=True):
        super(Dpam, self).__init__()
        
        self.layers = self.make_layer(planes_tab)

        self.softmax1 = nn.Softmax(dim=3)
        self.softmax2 = nn.Softmax(dim=2)
        self.eye = torch.eye(planes_tab[-1]).unsqueeze(0).unsqueeze(0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if not learnable:
            for p in self.parameters():
                p.requires_grad = False
    
    def make_layer(self, planes_tab):
        
        layers = nn.ModuleList()
        for i in range(len(planes_tab) - 1):
            layers.append(nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(planes_tab[i], planes_tab[i + 1], kernel_size=(1, 1), bias=False)), 
                    ('bn', nn.BatchNorm2d(planes_tab[i + 1])), 
                    ('relu', nn.ReLU(inplace=True))])))

        return layers

    def forward(self, x):
        
        self.eye = self.eye.to(x)
        
        if (~torch.isfinite(x)).sum().item() != 0:
            raise Exception('NaN or inf in x')
        
        x = x.view(x.size(0), 1, x.size(1), -1)
        # x: torch.FloatTensor[B, 1, C, N]
        
        # Eq. (4)
        # Is the softmax in Eq. (4) row-wise or col-wise?
        A = self.softmax1(torch.matmul(x.transpose(2, 3), x))
        # A: torch.FloatTensor[B, 1, N, N]
        
        # layers in ModuleList
        for i in range(0, len(self.layers)):
            if i < len(self.layers) - 2:
                x = torch.matmul(x, A)
            x = self.layers[i](x.transpose(1, 2)).transpose(1, 2)

        # Eq. (5)
        x = self.softmax2(x)
        
        # Eq. (6)
        l = (self.eye - torch.matmul(x, x.transpose(2, 3))).view(x.size(0), -1).norm(dim=1)
        # l: torch.FloatTensor[B]

        return x, l
    
class Mlp(nn.Module):
    
    def __init__(self, planes_tab=[], learnable=True):
        super(Mlp, self).__init__()
        
        self.layers = self.make_layer(planes_tab)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if not learnable:
            for p in self.parameters():
                p.requires_grad = False
                
    def make_layer(self, planes_tab):
        
        layers = nn.ModuleList()
        for i in range(len(planes_tab) - 1):
            layers.append(nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(planes_tab[i], planes_tab[i + 1], kernel_size=(1, 1), bias=False)), 
                    ('bn', nn.BatchNorm2d(planes_tab[i + 1])), 
                    ('relu', nn.ReLU(inplace=True))])))

        return layers
    
    def forward(self, x):

        # layers in ModuleList
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        return x
    
class DpamNet(nn.Module):
    
    def __init__(self, num_classes):
        super(DpamNet, self).__init__()
        
        self.mlp1 = Mlp(planes_tab=[3, 64, 128])
        self.mlp2 = Mlp(planes_tab=[128, 128])
        self.mlp3 = Mlp(planes_tab=[128, 256])
        self.mlp4 = Mlp(planes_tab=[256, 1024])
        
        self.dpam1 = Dpam(planes_tab=[128, 128, 256, 512, 128, 512])
        self.dpam2 = Dpam(planes_tab=[128, 128, 256, 512, 128, 256])
        self.dpam3 = Dpam(planes_tab=[256, 128, 256, 512, 128, 128])
        
        self.fcdown = nn.Linear(1536, 512, bias=False)
        self.fcfeat = nn.Linear(512, 256, bias=False)
        self.fc = nn.Linear(256, num_classes, bias=False)
        
        self.drop1 = nn.Dropout(0.7)
        self.drop2 = nn.Dropout(0.7)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, **kwargs):
        
        # x: torch.FloatTensor[B, C, N, 1]
        
        # otype: class/feat/image
        otype = kwargs.get('otype', 'class')
        
        x = self.mlp1(x[:, 0:3, ...])
        f1, _ = x.max(dim=2)
        s, l1 = self.dpam1(x)
        # s: torch.FloatTensor[B, 1, 512, 1024]
        # x: torch.FloatTensor[B, 128, 1024, 1]
        x = torch.matmul(s, x)
        # x: torch.FloatTensor[B, 128, 512, 1]
        
        x = self.mlp2(x)
        f2, _ = x.max(dim=2)
        s, l2 = self.dpam2(x)
        # s: torch.FloatTensor[B, 1, 256, 512]
        # x: torch.FloatTensor[B, 128, 512, 1]
        x = torch.matmul(s, x)
        # x: torch.FloatTensor[B, 128, 256, 1]
        
        x = self.mlp3(x)
        f3, _ = x.max(dim=2)
        s, l3 = self.dpam3(x)
        # s: torch.FloatTensor[B, 1, 128, 256]
        # x: torch.FloatTensor[B, 256, 256, 1]
        x = torch.matmul(s, x)
        # x: torch.FloatTensor[B, 256, 128, 1]
        
        x = self.mlp4(x)
        # x: torch.FloatTensor[B, 1024, 128, 1]
        f4, _ = x.max(dim=2)
        
        # f1: torch.FloatTensor[B, 128, 1]
        # f2: torch.FloatTensor[B, 128, 1]
        # f3: torch.FloatTensor[B, 256, 1]
        # f4: torch.FloatTensor[B, 1024, 1]
        
        if otype == 'class' or otype == 'feat':
            x = torch.cat((f1, f2, f3, f4), 1).reshape(x.size(0), -1)
            # x: torch.FloatTensor[B, 1536]
            
            x = self.fcdown(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.drop1(x)
            x = self.fcfeat(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.drop2(x)
        
        if otype == 'class':
            x = self.fc(x)
        
        l = l1 + l2 + l3
        # l: torch.FloatTensor[B]
        
        return x, l
