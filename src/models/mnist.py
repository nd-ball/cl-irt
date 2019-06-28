
from __future__ import print_function
import argparse
import collections
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data.my_data_downloaders import my_MNIST

from features.build_features import get_epoch_training_data_vision


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_data, test_loader, 
            optimizer, epoch, best_acc):
    model.train()
    imageIDs = []
    targets = []
    preds = []

    train_loader = get_epoch_training_data_vision(train_data, args, epoch) 
    for batch_idx, (data, target, label, diff, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        imageIDs.extend(label)
        targets.extend(target)
        preds.extend(output)

    model.eval()
    test_loss = 0
    correct = 0
    test_imageIDs = []
    test_targets = []
    test_preds = []

    with torch.no_grad():
        for data, target, label, diff, _ in test_loader:
            data, target, label = data.to(device), target.to(device), label.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_imageIDs.extend(label)
            test_targets.extend(target)
            test_preds.extend(pred)


    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    epoch_test_acc = 100. * correct / len(test_loader.dataset)
    if epoch_test_acc > best_acc:
        print('best test acc: {}, (epoch {})'.format(
            epoch_test_acc, epoch
        ))
        best_acc = epoch_test_acc

    return best_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', help='path to SNLI dataset')
    parser.add_argument('--num-units', type=int, default=300, help='number of units per layer')
    parser.add_argument('--balanced', action='store_true') 
    parser.add_argument('--strategy', choices=['baseline', 'ordered', 'simple'],
                        help='CL data policy', default='simple')
    parser.add_argument('--ordering', choices=['easiest', 'hardest', 'middleout'], default='easiest') 
    parser.add_argument('--num-epochs', type=int, default=50) 
    parser.add_argument('--random', action='store_true') 


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    mnist_train = my_MNIST(
        args.data_dir + '/external/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        diff_dir = args.data_dir + '/raw/' 
    )

    if len(mnist_train) == 0:
        print('no training data\n-1')
        return

    mnist_test = my_MNIST(
        args.data_dir + '/external/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        diff_dir = args.data_dir + '/raw/'
    )

    test_loader = torch.utils.data.DataLoader(mnist_test,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    best_test = 0

    for epoch in range(1, args.num_epochs + 1):             
        best_test = train(args, model, device, mnist_train, test_loader, optimizer, epoch, best_test)
    last_line = '{}'.format(best_test)
    print(last_line)

if __name__ == '__main__':
    main()
