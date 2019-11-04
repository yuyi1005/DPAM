# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:43:47 2019

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import dataset
import ext_scheduler
import dpamnet
import argparse

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, save_filename, num_epochs=50):
    since = time.time()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            itera_loss = 0.0
            itera_loss_d = 0.0
            itera_corrects = [0.0, 0.0, 0.0, 0.0, 0.0]
            itera_cnt = 0
            batch_cnt = 0
            time_load = 0
            time_forward = 0
            time_backward = 0

            # Iterate over data.
            a = time.time()
            for inputs, labels in dataloaders[phase]:
                
                labels_cpu = labels
                inputs = inputs.to(device)
                labels = labels.to(device)
                b = time.time()
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    try:
                        outputs, loss_d = model(inputs, label=labels)
                    except:
                        print(inputs.size(), labels_cpu)
                        raise
                    
                    loss_d = loss_d.mean()
                    loss = criterion(outputs, labels) + 0.001 * loss_d
                    c = time.time()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        torch.cuda.synchronize()
                    d = time.time()
                    
                _, idx = torch.sort(outputs, 1, descending=True)
                    
                preds = []
                for i in range(5):
                    preds.append(idx[:, i])

                # statistics
                itera_loss += loss.item() * inputs.size(0)
                itera_loss_d += loss_d.item() * inputs.size(0)
                for i in range(5):
                    itera_corrects[i] += torch.sum(preds[i] == labels.data).item()
                
                itera_cnt += 1
                batch_cnt += inputs.size(0)
                
                if itera_cnt % 10 == 0:
                    print('.', end='', flush=True)
                if itera_cnt % 100 == 0:
                    print('({}/{:.4f}/{:.4f})'.format(itera_cnt, itera_loss / batch_cnt, itera_loss_d / batch_cnt), end='', flush=True)
                    
                if phase == 'train':
                    with open(save_filename + '.loss.txt', mode='a') as f:
                        f.write('{:.4f} '.format(loss.item()))
                    
                time_load += b - a
                time_forward += c - b
                time_backward += d - c
                a = time.time()

            if itera_cnt == 0:
                raise Exception('Not enough samples')
            
            epoch_loss = itera_loss / batch_cnt
            epoch_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
            for i in range(5):
                epoch_acc[i] = itera_corrects[i] / batch_cnt

            epoch_sum = '{} Loss: {:.4f} Rank1: {:.4f} Rank2: {:.4f} Rank3: {:.4f} Rank4: {:.4f} Rank5: {:.4f}'.format(
                    phase, epoch_loss, 
                    epoch_acc[0],
                    epoch_acc[0] + epoch_acc[1],
                    epoch_acc[0] + epoch_acc[1] + epoch_acc[2],
                    epoch_acc[0] + epoch_acc[1] + epoch_acc[2] + epoch_acc[3],
                    epoch_acc[0] + epoch_acc[1] + epoch_acc[2] + epoch_acc[3] + epoch_acc[4])
            epoch_tim = 'Load: {:.3f} Forward: {:.3f} Backward: {:.3f}'.format(
                    time_load / itera_cnt, time_forward / itera_cnt, time_backward / itera_cnt)

            print()
            print(epoch_sum)
            print(epoch_tim)
            
            with open(save_filename + '.iter.txt', mode='a') as f:
                f.write('{} {}\n'.format(epoch_sum, epoch_tim))

            # save the model
            if phase == 'train':
                if isinstance(scheduler, tuple):
                    for s in scheduler:
                        s.step(epoch=epoch)
                else:
                    scheduler.step(epoch=epoch)
                torch.save(model.state_dict(), save_filename)
            if phase == 'val':
                if epoch_acc[0] > best_acc:
                    best_acc = epoch_acc[0]
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_filename.replace('.', '.best.', 1))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, Best epoch: {}'.format(best_acc, best_epoch))

    return model

def parse_args():
    
    parser = argparse.ArgumentParser(description="Arguments for training", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data_dir", type=str, default=R'H:\ModelNet40', help="Dataset directory")
    parser.add_argument("--load_filename", type=str, default=None)
    parser.add_argument("--save_filename", type=str, default='dpamnet1104.pt')
    parser.add_argument("--rmap_filename", type=str, default=None, help="Remap file in loading checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--point_num_max", type=int, default=1024)

    return parser.parse_args()

if __name__=="__main__":
    print('Pytorch version: ', torch.__version__)
    
    args = parse_args()
    
    data_dir = args.data_dir
    load_filename = args.load_filename
    save_filename = args.save_filename
    rmap_filename = args.rmap_filename
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    point_num_max = args.point_num_max
    
    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Create the model and send the model to GPU
    model = dpamnet.DpamNet(num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    if load_filename is not None:
        # Load the model from file
        model_dict = model.state_dict()
        saved_dict = torch.load(load_filename)
            
        if rmap_filename is not None:
            with open(rmap_filename, mode='r') as f:
                loadmap = f.readlines()
            for s in loadmap:
                k = s.split()
                saved_dict[k[1]] = saved_dict.pop(k[0])
        
        model_dict.update(saved_dict)
        model.load_state_dict(model_dict)
    
    # Create training and validation datasets, as well as dataloaders
    train_datasets = {x:dataset.BcDataset(os.path.join(data_dir, x), num_classes, point_num_max=point_num_max, use_transform=True) for x in ['train', 'val']}
    train_dataloaders = {x:torch.utils.data.DataLoader(train_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'val']}
    print('Num train/val:', len(train_datasets['train']), len(train_datasets['val']))
    
    # Criterion, optimizer, and schedulers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = (optim.lr_scheduler.LambdaLR(optimizer, last_epoch=-1, 
                                             lr_lambda=(lambda e: max(0.5 ** (int(e / 20)), 1e-3))), 
                 ext_scheduler.BNMomentum(model, last_epoch=-1, 
                                          bn_lambda=(lambda e: max(0.5 * 0.5 ** (int(e / 20)), 1e-2))))
    
    # Train and evaluate
    model = train_model(model, train_dataloaders, criterion, optimizer, scheduler, device, save_filename=save_filename, num_epochs=num_epochs)
    
