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

from models.models import vgg 

from data.my_data_downloaders import my_CIFAR10

from features.build_features import get_epoch_training_data_vision

parser = argparse.ArgumentParser(description='PyTorch DNN Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--balanced', action='store_true') 
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--data-dir', help='path to SNLI dataset')
parser.add_argument('--num-units', type=int, default=300, help='number of units per layer')
parser.add_argument('--balanced', action='store_true') 
parser.add_argument('--strategy', choices=['baseline', 'ordered', 'simple'],
                    help='CL data policy', default='simple')
parser.add_argument('--ordering', choices=['easiest', 'hardest', 'middleout'], default='easiest') 
parser.add_argument('--num-epochs', type=int, default=50) 
parser.add_argument('--random', action='store_true') 

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Data

print('==> Preparing data..')
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


testset = my_CIFAR10(root=args.data_dir + '/external/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = vgg.VGG('VGG16')
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

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print('Training')
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

    trainloader = get_epoch_training_data_vision(trainset, args, epoch) 

    #target_counts = collections.Counter([m[1] for m in trainset])
    #print(target_counts)

    for batch_idx, (inputs, targets, label, _, _) in enumerate(trainloader):
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

    # testing
    print('Testing')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc
        print('epoch: {}, best acc: {}'.format(epoch, best_acc))
    return best_acc


for epoch in range(0, args.num_epochs):
    ba = train(epoch)
    #test(epoch)
print(ba) 
print(len(trainset))
print(target_counts)
