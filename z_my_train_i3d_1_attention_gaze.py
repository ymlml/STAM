from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms_attention


import numpy as np

from pytorch_i3d_attention_gaze import InceptionI3d

from z_my_egtea_dataset_attention_gaze import Egtea as Dataset

n_wkers = 2
b_sz = 12
mx_steps = 64e3

# try to set random seed
manualSeed = 922
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)


def run(init_lr=0.1, max_steps=64e3, mode='rgb', batch_size=b_sz, save_model=''):

    if mode == 'flow':
        root = './dataset/EGTEA_Gaze_plus/z_frames_flow'
    else:
        root = './dataset/EGTEA_Gaze_plus/z_frames_rgb'

    gaze_root = './dataset/EGTEA_Gaze_plus/z_gaze_data'
    train_split = './dataset/EGTEA_Gaze_plus/action_annotation/train_split1.txt'

    # setup dataset
    train_transforms = transforms.Compose([videotransforms_attention.RandomCrop(224),
                                           videotransforms_attention.RandomHorizontalFlip(),
    ])

    dataset = Dataset(train_split, 'training', root, gaze_root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_wkers, pin_memory=True)


    dataloaders = {'train': dataloader}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(106, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_o.pt'), strict=False)
    else:
        i3d = InceptionI3d(106, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_o.pt'), strict=False)

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print(optimizer.param_groups[0]['lr'])
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_gaze_loss = 0.0

            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels, gt_labels, gaze_locs = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                gt_labels = Variable(gt_labels.cuda())

                per_frame_logits, a_map = i3d(inputs)

                #print inputs.shape, per_frame_logits.shape, gaze_locs.shape #(12, 3, 16, 224, 224) (12, 106, 1) (12, 16, 2)

                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                loss = F.cross_entropy(torch.max(per_frame_logits, dim=2)[0], gt_labels)
                loss /= num_steps_per_update

                tot_loss += loss.data.item()
                #loss.backward()

                a_map = a_map.squeeze(1)

                map_N, map_t, map_height, map_width = a_map.shape
                map_Data_s = torch.zeros([map_N, map_t, map_height, map_width], dtype=torch.float32)

                for n_idx in range(map_N):
                    gaze_avg1 = torch.mean(gaze_locs[n_idx, 0:8, :], dim=0)
                    gaze_avg2 = torch.mean(gaze_locs[n_idx, 8:16, :], dim=0)

                    x, y = gaze_avg1[0] * map_width, gaze_avg1[1] * map_height
                    u, v = gaze_avg2[0] * map_width, gaze_avg2[1] * map_height
                    sigma = 3
                    for i in range(0, map_height):
                        for j in range(0, map_width):
                            map_Data_s[n_idx, 0, i, j] = torch.exp(-1.0 * ((j - x) * (j - x) + (i - y) * (i - y)) / (2 * sigma * sigma))
                            map_Data_s[n_idx, 1, i, j] = torch.exp(-1.0 * ((j - u) * (j - u) + (i - v) * (i - v)) / (2 * sigma * sigma))

                map_Data_s = map_Data_s.cuda()

                gaze_loss = F.mse_loss(map_Data_s, a_map)
                gaze_loss /= num_steps_per_update

                tot_gaze_loss += gaze_loss.item()

                loss_two = loss + gaze_loss*0.1
                loss_two.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('{} Tot Loss: {:.4f} Gaze Loss: {:.4f} '.format(phase, tot_loss/10, tot_gaze_loss/10))
                        if steps % 1000 == 0:
                            # save model
                            torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_gaze_loss = 0.



if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, save_model=args.save_model, max_steps=mx_steps)
