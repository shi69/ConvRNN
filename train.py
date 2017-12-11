import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm

from dataset import get_ucf101
from model import Net

# arguments
parser = argparse.ArgumentParser(description='PyTorch CRNN')
parser.add_argument('--save_dir', default='results')
parser.add_argument('--batchSize', type=int, default=10)
parser.add_argument('--seq_len', type=int, default=20)
parser.add_argument('--grad_norm', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--arch', default='alexnet')
parser.add_argument('--use_rnn', default=True)
parser.add_argument('--resume', default=False)
parser.add_argument('--decay', type=float, default=0.1)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--multigpu', default=False)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--seed', type=int, default=123)
opt = parser.parse_args()
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    print('No GPU found. Switch to CPU mode.')
    cuda = False

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
trainset = get_ucf101('train', opt.seq_len, video_dir='../CRNN/frames')
testset = get_ucf101('test', opt.seq_len, video_dir='../CRNN/frames')
train_loader = DataLoader(dataset=trainset, num_workers=opt.threads,
                          batch_size=opt.batchSize, shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(dataset=testset, num_workers=opt.threads,
                         batch_size=opt.batchSize, shuffle=False,
                         pin_memory=True)
nbclasses = 101

print('===> Building model')
model = Net(opt.arch, nbclasses, opt.dropout)
print(model)
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = optim.Adam(model.params, lr=opt.lr)
optimizer.zero_grad()

def train(epoch):
    state = model.init_state(opt.batchSize)
    if cuda: state = [x.cuda() for x in state]
    model.train()
    epoch_loss = 0.
    for iteration, batch in enumerate(train_loader, 1):
        inputs, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            inputs = inputs.cuda()
            target = target.cuda()
       
        batch_loss, loss = 0, 0
        for t in range(opt.seq_len): # loop for time
            prob, state, gate = model(inputs[t], state, opt.use_rnn)
            loss += criterion(prob, target)
            batch_loss += loss.data[0] / opt.seq_len
            epoch_loss += loss.data[0] / opt.seq_len

        loss.backward()
        clip_grad_norm(model.params, opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # detaching history
        state = [Variable(x.data) for x in state]

        if iteration % opt.print_every == 0:
            print("===> Epoch[{}]({}/{}): Avg.Batch.Loss: {:.4f}".
                    format(epoch, iteration, len(train_loader), batch_loss))
    del state
    print("===> Epoch {} Complete: Avg.Epoch.Loss: {:.4f}".
            format(epoch, epoch_loss / len(train_loader)))


def evaluate(topk=(1,5)):
    print("===> Evaluating ...")
    state = model.init_state(opt.batchSize)
    if cuda: state = [x.cuda() for x in state]
    model.eval()
    epoch_loss = 0.
    prec = [0. for _ in topk]
    for iteration, batch in enumerate(test_loader):
        inputs, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            inputs = inputs.cuda()
            target = target.cuda()

        for t in range(opt.seq_len):
            prob, state, gate = model(inputs[t], state, opt.use_rnn)
            # detach history every batch
            state = [Variable(x.data) for x in state]
        loss = criterion(prob, target)
        epoch_loss += loss.data[0]
        res = accuracy(prob.data, target.data, topk)
        prec = [prec[i] + res[i] for i in range(len(topk))]

    epoch_loss = epoch_loss / len(test_loader)
    prec = [prec[i] / len(test_loader) for i in range(len(topk))]

    del state
    print("===> Avg.Epoch.Loss: {:.4f}, Avg.Prec@1: {:.2f}, Avg.Prec@5: {:.2f}".
            format(epoch_loss, prec[0], prec[1]))
    return epoch_loss, prec


def checkpoint(state, op='save'):
    """save the current training status to file"""
    filename = '{}_rnn{:d}_dropout{:.1f}.pth'.format(opt.arch, opt.use_rnn, opt.dropout)
    savepath = os.path.join(opt.save_dir, filename)
    if op == 'save':
        torch.save(state, savepath)
        print("Checkpoint saved to {}".format(savepath))
    elif op == 'load':
        return torch.load(savepath, map_location=lambda storage, loc: storage)


def accuracy(output, target, topk):
    """compute classification precision"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / opt.batchSize))
    return [x[0] for x in res]


# MAIN
flag = 0 # flag to decide next step
init_epoch = 1
prev_loss = 1e6
if opt.resume:
    gstate = checkpoint([], 'load')
    model.load_state_dict(gstate['model'])
    init_epoch = gstate['epoch']
    prev_loss = gstate['loss']
    prev_prec = gstate['prec'][0]
    print('===> Resume from prec {:.2f}'.format(prev_prec))

if cuda:
    if opt.multigpu:
        model = nn.DataParallel(model)
    model = model.cuda()
    criterion = criterion.cuda()

for epoch in range(init_epoch, opt.epochs):
    train(epoch)
    loss, prec = evaluate(topk=(1,5))

    # check stop criteria
    if loss <= prev_loss:
        flag = 0
        prev_loss = loss
        checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'loss': loss,
            'prec': prec,
            }, op='save')
    else:
        flag += 1
        if flag == opt.patience:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * opt.decay
                print("===> Learning Rate: {:.2e}".format(param_group['lr']))
        elif flag > opt.patience:
            print("===> Training stopped.")
            break
