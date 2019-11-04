# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:48:48 2019

@author: Administrator
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import math
import io3d

class BcDataset(Dataset):

    def __init__(self, path, class_num_max, point_num_max=0, use_transform=False):
        
        classlist = os.listdir(path)
        self.filelist = []
        self.labellist = []
        for i in range(min(len(classlist), class_num_max)):
            classpath = os.path.join(path, classlist[i])
            classfilelist = os.listdir(classpath)
            for j in range(len(classfilelist)):
                self.filelist.append(os.path.join(classpath, classfilelist[j]))
                self.labellist.append(i)
        
        self.point_num_max = point_num_max
        self.use_transform = use_transform
        self.x_dummy = torch.cat((90 * torch.ones(3, 1, 1), torch.zeros(2, 1, 1), torch.ones(1, 1, 1), torch.zeros(1, 1, 1)), 0)
    
    def vector2dcm(self, p1, p2):
        #Get the matrix R that satisfies R*p1=p2
        #p1,p2 are tensors of 3 elements
        
        theta = math.acos(torch.sum(p1 * p2))
        p3 = p1.clone()
        p3[0] = p1[1] * p2[2] - p1[2] * p2[1]
        p3[1] = p1[2] * p2[0] - p1[0] * p2[2]
        p3[2] = p1[0] * p2[1] - p1[1] * p2[0]
        p3n = p3.norm()
        if p3n != 0:
            p3 = p3 / p3n
        
        r0 = torch.tensor(((0, -p3[2], p3[1]), 
                           (p3[2], 0, -p3[0]),
                           (-p3[1], p3[0], 0)))
        #Rodrigues
        R = math.cos(theta) * torch.eye(3, 3) + (1 - math.cos(theta)) * p3.unsqueeze(1).mm(p3.unsqueeze(0)) + math.sin(theta) * r0
    
        return R
    
    def transform_rt(self, x, r=None, t=None):
        
        if r is None and t is None:
            p = x[0:3, :, :]
        elif r is None:
            p = (x[0:3, :, :].permute(2, 1, 0) + t).permute(2, 1, 0)
        elif t is None:
            p = (torch.matmul(x[0:3, :, :].permute(2, 1, 0), r.t())).permute(2, 1, 0)
        else:
            p = (torch.matmul(x[0:3, :, :].permute(2, 1, 0), r.t()) + t).permute(2, 1, 0)
            
        if x.size(0) >= 6:
            if r is None:
                n = x[3:6, :, :]
            else:
                n = (torch.matmul(x[3:6, :, :].permute(2, 1, 0), r.t())).permute(2, 1, 0)
            
        return torch.cat((p, n, x[6:, :, :]), 0)
    
    def random_rotate(self, x):
        
        vector1 = torch.from_numpy(np.float32(np.random.rand(3) - 0.5)) * 0.5 + torch.tensor((0., 0, 1))
        vector1 /= vector1.norm()
        vector2 = torch.from_numpy(np.float32(np.random.rand(3) - 0.5)) * 0.5 + torch.tensor((0., 0, 1))
        vector2 /= vector2.norm()
        
        x = self.transform_rt(x, self.vector2dcm(vector1, vector2))
        
        return x
    
    def __getitem__(self, index):
        
        path = self.filelist[index]
        if path.endswith('bc'):
            pc = io3d.loadBCFile(path)
            x = torch.from_numpy(pc).view(3, -1, 1)
        elif path.endswith('bcn'):
            pc = io3d.loadBCNFile(path)
            x = torch.from_numpy(pc).view(6, -1, 1)
        elif path.endswith('bcnc'):
            pc = io3d.loadBCNCFile(path)
            x = torch.from_numpy(pc).view(7, -1, 1)
        
        if self.point_num_max > 0:
            if x.size(1) >= self.point_num_max:
                idx = np.random.choice(x.size(1), self.point_num_max, replace=False)
                x = x[:, idx, :]
                if self.use_transform:
                    x = self.random_rotate(x)
            else:
                idx = np.random.choice(x.size(1), x.size(1), replace=False)
                x = x[:, idx, :]
                if self.use_transform:
                    x = self.random_rotate(x)
                x = torch.cat((x, self.x_dummy[0:x.size(0), ...].repeat(1, self.point_num_max - x.size(1), 1)), 1)
        
        return x, self.labellist[index]

    def __len__(self):
        return len(self.filelist)
