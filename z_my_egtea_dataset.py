import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2

# Set random seem for reproducibility
manualSeed = 922
print("Random Seed: ", manualSeed)
random.seed(manualSeed)


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    #img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    img = cv2.imread(os.path.join(vid, str(i).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
    if img.shape[0] != 240 or img.shape[1] != 320:
        img = cv2.resize(img, dsize=(320, 240))  # --> 320x240 if needed

    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)

    img = (img/255.)*2 - 1
    frames.append(img)

  return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    #imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    #imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    imgx = cv2.imread(os.path.join(vid, str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(vid, str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE) #flow images are 320x240

    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=106):
    dataset = []

    with open(split_file, 'r') as f:
        lines = f.readlines()

    for i, l in enumerate(lines):
        sp = l.split(' ')

        vid = sp[0]
        gt_label = int(sp[1]) - 1  # 1-106 ==> 0-105

        spp = vid.split('-')
        vid_folder = '-'.join(spp[0:3])

        vid_path = os.path.join(root, vid_folder, vid)
        if not os.path.exists( vid_path ):
            print 'no video input'
            continue
        num_frames = len(os.listdir( vid_path ))

        if mode == 'flow':
            num_frames = num_frames//2

        if num_frames < 16:
            print 'input video too short'
            continue

        label = np.zeros((num_classes,num_frames), np.float32)
        label[ gt_label, :] = 1

        dataset.append((vid_path, label, gt_label, num_frames))

    return dataset



class Egtea(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.my_clip_len = 16

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid_path, label, gt_label, nf = self.data[index]
        start_f = random.randint(0, nf-self.my_clip_len)  #

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid_path, start_f+1, self.my_clip_len)  #frame index starts from 1
        else:
            imgs = load_flow_frames(self.root, vid_path, start_f+1, self.my_clip_len)
        label = label[:, start_f:start_f+self.my_clip_len]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), gt_label

    def __len__(self):
        return len(self.data)
