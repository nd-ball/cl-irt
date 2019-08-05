'''
Train a model with randomly sampled data and noise,
and write outputs to disk
'''
from __future__ import print_function

import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import csv
import os

from data.my_data_downloaders import my_CIFAR10

from features.build_features import get_epoch_training_data_vision, k_sort 
from features.irt_scoring import calculate_theta 


### Define VGG model
'''VGG11/13/16/19 in Pytorch.'''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



parser = argparse.ArgumentParser(description='PyTorch DNN Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--data-dir', help='path to SNLI dataset')
parser.add_argument('--num-units', type=int, default=300, help='number of units per layer')
parser.add_argument('--balanced', action='store_true') 
parser.add_argument('--strategy', choices=['baseline', 'ordered', 'simple', 'theta', 'naacl-linear', 'naacl-root', 'theta-hard'],
                    help='CL data policy', default='simple')
parser.add_argument('--ordering', choices=['easiest', 'hardest', 'middleout'], default='easiest') 
parser.add_argument('--num-epochs', type=int, default=50) 
parser.add_argument('--random', action='store_true') 
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--min-train-length', default=100, type=int)
parser.add_argument('--k', default=0, type=int) 
parser.add_argument('--competency', default=50, type=int) 
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

preds_file = '{}processed/test_predictions/cifar_{}_{}_{}_{}_{}.csv'.format(args.data_dir, args.strategy, args.balanced, args.ordering, args.random, args.k) 
outfile = open(preds_file, 'w') 
outwriter = csv.writer(outfile, delimiter=',')
outwriter.writerow(['epoch', 'itemID', 'correct', 'pred'])

# Data

#print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = my_CIFAR10(root=args.data_dir + '/external/', train=True, download=True, 
                    transform=transform_train, diff_dir=args.data_dir + '/raw/')

valset = [trainset[i] for i in range(40000,50000)]
trainset = [trainset[i] for i in range(40000)]

valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

testset = my_CIFAR10(root=args.data_dir + '/external/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

if args.strategy == 'theta':
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    irt_trainloader = torch.utils.data.DataLoader(valset,
                batch_size=args.batch_size, shuffle=False, **kwargs) 


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
#print('==> Building model..')
net = VGG('VGG16')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# if k is set, sort once
if args.k > 0:
    diffs = [img[3] for img in trainset]
    diffs_sorted_idx = k_sort(diffs, args.k) 
else:
    diffs_sorted_idx = None 

    
# Training
def train(epoch, outwriter, best_test, best_val):
    #print('\nEpoch: %d' % epoch)
    #print('Training')

    if args.strategy == 'theta':
        # get a theta estimate from entire training set 
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        train_diffs = [] 
        train_rps = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, label, diffs, _) in enumerate(irt_trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                rp = predicted.eq(targets).cpu().numpy() 
                train_diffs.extend(diffs.numpy()) 
                train_rps.extend(rp)
        # calculate theta based on current epoch data 
        train_rps = [j if j==1 else -1 for j in train_rps] 
        #print(train_diffs) 
        #print(train_rps) 
        theta_hat = calculate_theta(train_diffs, train_rps)[0] 
        #print('estimated theta: {}'.format(theta_hat)) 
    else:
        theta_hat = 0
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_labels = []
    train_targets = []
    train_preds = []
    test_labels = []
    test_targets = []
    test_preds = []
    val_labels = []
    val_targets = []
    val_preds = []

    trainloader = get_epoch_training_data_vision(trainset, args, epoch, theta_hat, diffs_sorted_idx)
    train_length = len(trainloader.dataset) 

    #target_counts = collections.Counter([m[1] for m in trainset])
    #print(target_counts)

    for batch_idx, (inputs, targets, label, diffs, _) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_labels.extend(label)
        train_targets.extend(targets)
        train_preds.extend(predicted)
    train_acc = 100. * correct / train_length  


    # testing
    #print('Testing')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, label, _, _) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            val_labels.extend(label)
            val_targets.extend(targets)
            val_preds.extend(predicted)
        val_loss /= len(testloader.dataset)

        # Save checkpoint.
        val_acc = 100.*val_correct/val_total
        for batch_idx, (inputs, targets, label, _, _) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_labels.extend(label)
            test_targets.extend(targets)
            test_preds.extend(predicted)
        test_loss /= len(testloader.dataset)

    # Save checkpoint.
    acc = 100.*correct/total
    print('{},{},{},{},{}'.format(train_length, train_acc, val_acc, acc, theta_hat))

    # write test predictions to file
    for i in range(len(test_preds)):
        row = [epoch, test_labels[i].cpu().item(), test_targets[i].cpu().item(), test_preds[i].cpu().item()]
        outwriter.writerow(row) 

    if val_acc > best_val:
        #print('Saving..')
        best_val = val_acc
        best_test = acc 
        #print('epoch: {}, best acc: {}'.format(epoch, best_acc))
    return best_test, best_val 

best_test, best_val = 0, 0
for epoch in range(0, args.num_epochs):
    best_test, best_val = train(epoch, outwriter, best_test, best_val)
    #test(epoch)
print(best_val, best_test) 
#print(len(trainset))
#print(target_counts)
