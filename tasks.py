import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from models import VGG, ResNet18
import random

default_datapath = '/tmp/data'
if 'SLURM_TMPDIR' in os.environ:
    default_datapath = os.path.join(os.environ['SLURM_TMPDIR'], 'data')

def to_tensordataset(dataset):
    d = next(iter(DataLoader(dataset,
                  batch_size=len(dataset))))
    return TensorDataset(d[0].to('cuda'), d[1].to('cuda'))

def extract_small_loader(baseloader, length, batch_size):
    datas = []
    targets = []
    i = 0
    for d, t in iter(baseloader):
        datas.append(d.to('cuda'))
        targets.append(t.to('cuda'))
        i += d.size(0)
        if i >= length:
            break
    datas = torch.cat(datas)[:length]
    targets = torch.cat(targets)[:length]
    dataset = TensorDataset(datas, targets)

    return DataLoader(dataset, shuffle=False, batch_size=batch_size)

def kaiming_init(net, tanh=False):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if tanh:
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

def get_cifar10(args):
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

    trainset = CIFAR10(root=default_datapath, train=True, download=True,
                       transform=transform_train)
    if args.diff_type == 'random' and args.diff > 0.:
        for i in range(len(trainset.targets)):
            if random.random() < args.diff:
                trainset.targets[i] = random.randint(0, 9)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = CIFAR10(root=default_datapath, train=False, download=True,
                      transform=transform_test)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    return trainloader, testloader

def get_task(args):
    dataloaders = dict()

    task_name, model_name = args.task.split('_')

    if task_name == 'cifar10':
        dataloaders['train'], dataloaders['test'] = get_cifar10(args)
        if model_name == 'vgg19':
            model = VGG('VGG19')
        elif model_name == 'resnet18':
            model = ResNet18

    model = model.to('cuda')
    kaiming_init(model)

    criterion = nn.CrossEntropyLoss()

    if args.align_train or args.layer_align_train or args.save_ntk_train:
        dataloaders['micro_train'] = extract_small_loader(dataloaders['train'], 100, 100)
    if args.align_test or args.layer_align_test or args.save_ntk_test:
        dataloaders['micro_test'] = extract_small_loader(dataloaders['test'], 100, 100)
    dataloaders['mini_test'] = extract_small_loader(dataloaders['test'], 1000, 1000)

    return model, dataloaders, criterion