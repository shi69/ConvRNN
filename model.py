import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from math import sqrt

class Net(nn.Module):
    """Recurrent Model"""
    def __init__(self, arch, nbclasses, dropout=0.5):
        super(Net, self).__init__()

        self.arch = arch
        self.nbclasses = nbclasses

        # Load Pre-trained CNN and fix the parameters
        self.cnn = models.__dict__[arch.split('-')[0]](pretrained=True)
        self.cnn.zero_grad()
        for param in self.cnn.parameters():
            param.requires_grad = False

        if arch.startswith('alexnet'):
            self.depth = 13
            self.nlayers = 5
            self.layers = [1, 4, 7, 9, 11] # After ReLU
            self.size = [55, 27, 13, 13, 13]
            self.nchannels = [64, 192, 384, 256, 256]
            self.nfc = sum(self.nchannels)
            self.avgpool = nn.ModuleList([nn.AvgPool2d(x) for x in self.size])
            self.maxpool = nn.MaxPool2d(3, 2)
        elif arch.startswith('vgg16'):
            self.depth = 31
            self.nlayers = 4
            self.layers = [9, 16, 23, 30] # After Maxpool
            self.size = [56, 28, 14, 7]
            self.nchannels = [128, 256, 512, 512]
            self.nfc = sum(self.nchannels)
            self.avgpool = nn.ModuleList([nn.AvgPool2d(x) for x in self.size])
            self.maxpool = nn.MaxPool2d(2, 2)

        # Feature extractors
        self.features = []
        for l in self.layers:
            self.cnn.features[l].register_forward_hook(
                lambda m, i, o: self.features.append(Variable(o.data)))

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else None

        # The list of active parameters
        self.params = nn.ParameterList()

        # Add Batch Normalization?
        if arch.endswith('bn'):
            self.bn = nn.ModuleList([nn.BatchNorm2d(x, eps=1e-3) for x in self.nchannels])
            for bn in self.bn:
                bn.weight.data.fill_(0.01)
                bn.bias.data.fill_(0.0)
                self.params.append(bn.weight)
                self.params.append(bn.bias)

        # Forget Gate
        self.g = nn.ModuleList()

        # Linear Classifier
        self.fc = nn.Linear(self.nfc, nbclasses)
        self.fc.weight.data.normal_(0.0, 1.0 / sqrt(self.nfc))
        self.fc.bias.data.fill_(0.0)
        self.params.append(self.fc.weight)
        self.params.append(self.fc.bias)

        for l in range(self.nlayers):
            if l == 0:
                self.g[l] = self._conv3x3(2 * self.nchannels[l], self.nchannels[l])
            else:
                self.g[l] = self._conv3x3(2 * self.nchannels[l] + self.nchannels[l-1], self.nchannels[l])
            self.params.append(self.g[l].weight)
            self.params.append(self.g[l].bias)
          
    def forward(self, input, state, use_rnn=True, extract_gate=False):
        _ = self.cnn.features(input) # extract features
        p = []
        gates = []
        num_batches = input.size(0)
        for l in range(self.nlayers):
            x = self.features[l] # current layer feature maps
            
            # Scale the feature maps to stablize RNN training
            if self.arch.endswith('bn'):
                x = self.bn[l](x)
            else:
                x = x / Variable(x.data.max(1)[0].max(1)[0].max(1)[0].view(nbatches,1,1,1).expand_as(x) + 1e-5, requires_grad=False)

            if use_rnn:
                # Seems like it can't automatically adjust the batch_size
                # at the end of every epoch where the remaining samples
                # are less than the batch_size
                if state[l].size(0) > x.size(0):
                    state[l] = state[l].narrow(0, 0, x.size(0))

                # Apply different Maxpooling Schemes based on the CNN
                if l == 0:
                    z = torch.cat([x, state[l]], 1)
                if l == 1 or l == 2:
                    z = torch.cat([x, state[l], self.maxpool(state[l-1])], 1)
                elif l > 2:
                    if self.arch.startswith('vgg16'):
                        z = torch.cat([x, state[l], self.maxpool(state[l-1]), 1)
                    else:
                        z = torch.cat([x, state[l], state[l-1], 1)
                   
                f = self.sigmoid(self.g[l](z))
                if extract_gate: gates.append(Variable(f.data))
                state[l] = (1.0 - f) * state[l] + f * x
                p.append(self.avgpool[l](state[l]).view(-1, self.nchannels[l]))

            else:
                p.append(self.avgpool[l](x).view(-1, self.nchannels[l]))

        p = torch.cat(p, 1)
        if self.dropout is not None: p = self.dropout(p)
        prob = self.fc(p)
        self.features = []
        return prob, state, gates

    def init_state(self, batch_size):
        state = []
        for l in range(self.nlayers):
            state.append(Variable(torch.zeros(batch_size, self.nchannels[l],
                                              self.size[l], self.size[l])))
        return state

    @staticmethod
    def _conv3x3(nin, nout, k=3, s=1, p=1, bias=True):
        conv = nn.Conv2d(nin, nout, k, s, p, bias=bias)
        conv.weight.data.uniform_(-0.01, 0.01)
        conv.bias.data.fill_(0.0)
        return conv
