from os import listdir
from os.path import join
import torch
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
import h5py
import numpy as np
import random

print('===> Building model')
use_rnn = True
arch = 'alexnet'
nbclasses = 101
dropout = 0.7

from model import Net
model = Net(arch, nbclasses, dropout)

filename = 'results/{}_rnn{}_dropout{:.1f}.pth.tar'.format(arch, use_rnn, dropout)
print('===> Loading checkpoint', filename)
gstate = torch.load(filename, map_location=lambda storage, loc: storage)
print('===> Prec@1: {:.2f}'.format(gstate['prec'][0]))
model.load_state_dict(gstate['model'])
model = model.cuda()
model.eval()

transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])


state = model.init_state(1) # batch_size = 1
state = [x.cuda() for x in state]

print('===> Processing training data')
for sec in range(1, 99): # 1 to 98 sections
    foldername = '/home/jun/HDD1/users/shi/data/movies/frames/section'+str(sec)
    myfile = h5py.File('/home/jun/HDD1/users/shi/data/movies/features/section'+str(sec)+'_'+arch+'_rnn.h5', 'w')
    grps = [myfile.create_group('layer'+str(l)) for l in range(1, model.nlayers+1)]
    print('Section', str(sec))
    for t in range(1, 2401):
        filename = join(foldername, 'img%04d.png' % (t))
        im = Image.open(filename)
        im = transform(im)
        im = Variable(im.view(1,3,224,224)).cuda()

        prob, state, gate = model(im, state, use_rnn)
        state = [Variable(x.data) for x in state]

        # save to file
        features = [x.data.squeeze(0).cpu().numpy() for x in state]

        for l in range(model.nlayers):
            grps[l].create_dataset('t'+str(t), data=features[l])

    myfile.close()
    state = [Variable(x.data.zero_()) for x in state]


img_folder = '/home/jun/HDD1/users/shi/data/movies/frames/'
target_folder = '/home/jun/HDD1/users/shi/data/movies/features/'

print('===> Processing testing data')
for sec in ['test', 'test3', 'test0', 'test1', 'test2']:
    foldername = img_folder+'section_'+sec

    myfile = h5py.File(target_folder+'section_'+sec+'_'+arch+'_rnn.h5', 'w')
    grps = [myfile.create_group('layer'+str(l)) for l in range(1, model.nlayers+1)]
    grps2 = [myfile.create_group('gate'+str(l)) for l in range(1, model.nlayers+1)]
    idx = np.linspace(1, 6720, 2400).round().astype(int)

    for t in range(2400):
        filename = join(foldername, '%d.jpg' % (idx[t]))
        im = Image.open(filename)
        im = transform(im)
        im = Variable(im.view(1,3,224,224)).cuda()

        prob, state, gate = model(im, state, use_rnn, extract_gate=True)
        state = [Variable(x.data) for x in state]

        # save to file
        features = [x.data.squeeze(0).cpu().numpy() for x in state]
        gate = [x.data.squeeze(0).cpu().numpy() for x in gate]
        for l in range(model.nlayers):
            grps[l].create_dataset('t'+str(t+1), data=features[l])
            grps2[l].create_dataset('t'+str(t+1), data=gate[l])

    myfile.close()
    state = [Variable(x.data.zero_()) for x in state]

