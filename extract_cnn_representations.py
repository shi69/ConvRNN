from os import listdir
from os.path import join
import torch
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
import h5py
import numpy as np

cnn = 'alexnet'
print('===> Building model')
model = models.alexnet(pretrained=True)

layers = [1, 4, 7, 9, 11] #layers = [9, 16, 23, 30]
features = []
for layer in layers:
    model.features[layer].register_forward_hook(
            lambda m, i, o: features.append(o))
model.features = model.features.cuda()
model = model.eval()

transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

def process(x):
    x.data.squeeze_(0)
    x.data.div_(x.data.max() + 1e-5)
    return x.data.cpu().numpy()

print('===> Processing training data')
for sec in range(1, 99): # 1 to 98 sections
    foldername = '/home/jun/HDD1/users/shi/data/movies/frames/section'+str(sec)
    myfile = h5py.File('/home/jun/HDD1/users/shi/data/movies/features/section'+str(sec)+'_'+cnn+'.h5', 'w')
    grps = [myfile.create_group('layer'+str(l)) for l in range(1, 1+len(layers))] # 1-5
    print('Section', str(sec))
    for t in range(1, 2401): # 1-2400
        filename = join(foldername, 'img%04d.png' % (t))
        im = Image.open(filename)
        im = transform(im)
        im = Variable(im.view(1,3,224,224)).cuda()

        _ = model.features(im)

        # save to file
        features = [process(x) for x in features]
        for l in range(len(features)):
            grps[l].create_dataset('t'+str(t), data=features[l])
        features = []
    myfile.close()


print('===> Processing testing data')
for sec in ['test', 'test0', 'test1', 'test2', 'test3']:
    foldername = '/home/jun/HDD1/users/shi/data/movies/frames/section_'+sec
    myfile = h5py.File('/home/jun/HDD1/users/shi/data/movies/features/section_'+sec+'_'+cnn+'.h5', 'w')
    grps = [myfile.create_group('layer'+str(l)) for l in range(1, 1+len(layers))] # 1-5
    idx = np.linspace(1, 6720, 2400).round().astype(int)
    print('Section', sec)
    for t in range(2400):
        filename = join(foldername, '%d.jpg' % (idx[t]))
        im = Image.open(filename)
        im = transform(im)
        im = Variable(im.view(1,3,224,224)).cuda()

        _ = model.features(im)

        # save to file
        features = [process(x) for x in features]
        for l in range(len(features)):
            grps[l].create_dataset('t'+str(t+1), data=features[l])
        features = []
    myfile.close()
