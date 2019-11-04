# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:20:18 2019

@author: Administrator
"""

import os
import struct
import numpy as np

def loadBCFile(path):
    n = os.path.getsize(path) // 4
    with open(path, 'rb') as f:
        data_raw = struct.unpack('f' * n, f.read(n * 4))
    data =  np.asarray(data_raw, dtype=np.float32).reshape(3, n // 3)
    return data

def loadBCNFile(path):
    n = os.path.getsize(path) // 4
    with open(path, 'rb') as f:
        data_raw = struct.unpack('f' * n, f.read(n * 4))
    data = np.asarray(data_raw, dtype=np.float32).reshape((3 + 3), n // (3 + 3))
    return data

def loadBCNCFile(path):
    n = os.path.getsize(path) // 4
    with open(path, 'rb') as f:
        data_raw = struct.unpack('f' * n, f.read(n * 4))
    data = np.asarray(data_raw, dtype=np.float32).reshape((3 + 3 + 1), n // (3 + 3 + 1))
    return data

def loadBCNKFile(path):
    n1 = os.path.getsize(path) // 4 // (3 + 3 + 8) * (3 + 3)
    n2 = os.path.getsize(path) // 4 // (3 + 3 + 8) * (8)
    with open(path, 'rb') as f:
        data_raw1 = struct.unpack('f' * n1, f.read(n1 * 4))
        data_raw2 = struct.unpack('i' * n2, f.read(n2 * 4))
    data1 = np.asarray(data_raw1, dtype=np.float32).reshape((3 + 3), n1 // (3 + 3))
    data2 = np.asarray(data_raw2, dtype=np.float32).reshape((8), n2 // (8)) * 100
    data = np.vstack((data1, data2))
    return data

def loadCOFFFile(path):
    n = os.path.getsize(path) // 4
    with open(path,'rb') as f:
        raw_data = struct.unpack('f' * n, f.read(n * 4))
    data = np.asarray(raw_data, dtype=np.float32)
    return data

def saveBCFile(path, data):
    if not data.shape[0] == 3:
        raise 'BC shape error'
    data_raw = data.reshape(-1)
    n = data_raw.shape[0]
    with open(path, 'wb') as f:
        f.write(struct.pack('f' * n, *data_raw))
        
def saveBCNFile(path, data):
    if not data.shape[0] == 6:
        raise 'BCN shape error'
    data_raw = data.reshape(-1)
    n = data_raw.shape[0]
    with open(path, 'wb') as f:
        f.write(struct.pack('f' * n, *data_raw))

def saveBCNCFile(path, data):
    if not data.shape[0] == 7:
        raise 'BCNC shape error'
    data_raw = data.reshape(-1)
    n = data_raw.shape[0]
    with open(path, 'wb') as f:
        f.write(struct.pack('f' * n, *data_raw))
