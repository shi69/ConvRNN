import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from os import listdir
from os.path import join
import numpy as np
from PIL import Image

def get_ucf101(kind, seq_len, video_dir, meta_dir='meta'):
    """
    The interface function. 
    Need to first prepare the video frames in the 'video_dir'.
    
    kind: 'train' or anything else
    """
    return DatasetFromTxt(kind, video_dir, meta_dir, seq_len,
                          input_transform=input_transform(kind))

def input_transform(phase):
    """Input transformation to Torch Tensor"""
    if phase.lower().startswith('train'):
        return transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])
    else:
        return transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

def load_image(filepath, seq_len):
    """The main image loading function"""
    filepath = filepath.split('.')[0]
    images = listdir(filepath)
    nimages = len(images)

    if_flip = np.random.random() < 0.5
    frames = []
    if nimages > seq_len:
        sindex = np.random.randint(0, nimages - seq_len + 1)
        eindex = sindex + seq_len
        reps = 0
    else:
        sindex = 0
        eindex = nimages
        reps = seq_len - nimages

    for rep in range(reps):
        x = Image.open(join(filepath, 'img0001.png'))
        if if_flip:
            x = x.transpose(Image.FLIP_LEFT_RIGHT)
        frames.append(x)

    for t in range(sindex, eindex):
        img_name = 'img%04d.png' % (t+1)
        x = Image.open(join(filepath, img_name))
        if if_flip:
            x = x.transpose(Image.FLIP_LEFT_RIGHT)
        frames.append(x)
    assert len(frames) == seq_len
    return frames

def load_txt(meta_dir, kind, video_dir):
    """Pre-loading training/validation data and labels"""
    meta = dict()
    target = dict()

    with open(join(meta_dir, 'classInd.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            ind, name = line.split('\n')[0].split(' ')
            target[name] = torch.from_numpy(np.array([int(ind)-1]))

    with open(join(meta_dir, kind+'list.txt'), 'r') as f:
        lines = f.readlines()
        meta['data'] = []
        meta['label'] = []
        for line in lines:
            if kind == 'train':
                name = line.split('\n')[0].split(' ')[0]
            else:
                name = line.split('\n')[0]

            class_name = name.split('/')[0]
            meta['data'].append(join(video_dir, name))
            meta['label'].append(target[class_name])

    assert len(meta['data']) == len(meta['label'])
    return meta

class DatasetFromTxt(data.Dataset):
    """The data loader object."""
    def __init__(self, kind, video_dir, meta_dir, seq_len, input_transform=None, target_transform=None):
        super(DatasetFromTxt, self).__init__()

        self.meta = load_txt(meta_dir, kind, video_dir)
        self.seq_len = seq_len
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        index %= len(self.meta['data'])
        inputs = load_image(self.meta['data'][index], self.seq_len)
        target = self.meta['label'][index]

        if self.input_transform:
            inputs = torch.stack([self.input_transform(input) for input in inputs], 3)
        if self.target_transform:
            target = self.target_transform(target)

        return inputs, target

    def __len__(self):
        return len(self.meta['label'])

