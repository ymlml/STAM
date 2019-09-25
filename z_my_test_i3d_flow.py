import cv2
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torchvision import datasets, transforms
import videotransforms

from pytorch_i3d import InceptionI3d

import time

root = './dataset/EGTEA_Gaze_plus/z_frames_flow'
test_split = './dataset/EGTEA_Gaze_plus/action_annotation/test_split1.txt'
#test_split = 'zzz.txt'

g_mode = 'flow'
g_num_classes = 106
g_offset = 8
g_clip_length = 16

start_time = time.time()

# setup the model
i3d = InceptionI3d(400, in_channels=2)
i3d.replace_logits(g_num_classes)
#i3d.load_state_dict(torch.load('tmp_flow/tmp064000.pt'))
i3d.load_state_dict(torch.load('models/flow_o.pt'))

i3d.cuda()
i3d = nn.DataParallel(i3d)

i3d.eval()

with open(test_split, 'r') as f:
    lines = f.readlines()

num_instances = len(lines)
print num_instances

test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

class_correct = np.zeros(g_num_classes)
class_total = np.zeros(g_num_classes)

all_scores = np.zeros((num_instances, g_num_classes))
gggg_labels = np.zeros(num_instances)


def load_video_frames(video_path, start, end):
    a_frames = []
    for ii in range(start, end):
        imgx = cv2.imread(os.path.join(video_path, str(ii).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(video_path, str(ii).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)  # flow images are 320x240

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])

        a_frames.append(img)

    if len(a_frames)<g_clip_length:
        tmp = a_frames[-1]
        a_frames.append( tmp )

    return a_frames


with torch.no_grad():

    cnt = 0

    for ixx, l in enumerate(lines):
        sp = l.split(' ')

        vid = sp[0]
        gt_label = int(sp[1]) - 1  # 1-106 ==> 0-105

        spp = vid.split('-')
        vid_folder = '-'.join(spp[0:3])

        vid_path = os.path.join(root, vid_folder, vid)
        if not os.path.exists(vid_path):
            print 'no video input'
            continue
        num_frames = len(os.listdir(vid_path))

        if g_mode == 'flow':
            num_frames = num_frames // 2

        print vid, num_frames

        if num_frames < g_clip_length:
            print 'input video too short'
            #continue

        all_frames = load_video_frames(vid_path, 1, num_frames+1)

        score = torch.zeros(g_num_classes).cuda()

        for idx in range(1, num_frames + 1, g_offset):
            if (idx + g_clip_length) <= num_frames + 1:

                frames = all_frames[idx : idx+g_clip_length]

                inputs = np.asarray(frames, dtype=np.float32)
                inputs = test_transforms(inputs)
                inputs = torch.from_numpy(inputs.transpose([3, 0, 1, 2]))
                inputs = inputs.unsqueeze(0)
                inputs = Variable(inputs.cuda())

                per_frame_logits = i3d(inputs)

                score = score + per_frame_logits[0, :, 0]

            else:

                frames = all_frames[num_frames+1-g_clip_length : num_frames+1]

                inputs = np.asarray(frames, dtype=np.float32)
                inputs = test_transforms(inputs)
                inputs = torch.from_numpy(inputs.transpose([3, 0, 1, 2]))
                inputs = inputs.unsqueeze(0)
                inputs = Variable(inputs.cuda())

                per_frame_logits = i3d(inputs)

                score = score + per_frame_logits[0, :, 0]


        if torch.max(score, dim=0)[1].item() == gt_label:
            cnt += 1
            class_correct[gt_label] += 1

        class_total[gt_label] += 1

        all_scores[ixx] = score.cpu().numpy()
        gggg_labels[ixx] = gt_label

        print torch.max(score, dim=0)[1].item(), gt_label
        print 'xxxxxxxxxxxxxxxx'

    print cnt, num_instances
    print sum(class_correct), sum(class_total)

    print 'Accuracy'
    print 1.0*cnt/num_instances

    print 'Mean Class Accuracy'
    print sum(class_correct/class_total) / g_num_classes

    #print class_correct
    #print class_total
    #print class_correct/class_total


end_time = time.time()
print 'time: ', end_time - start_time


np.savetxt('scores/flow.txt', all_scores)
np.savetxt('scores/gt_labels.txt', gggg_labels)