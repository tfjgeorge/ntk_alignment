import argparse
from tasks import get_task
import time
import os
import pandas as pd
import torch.optim as optim
import torch
from alignment import alignment, layer_alignment, compute_trK
import numpy as np
from nngeometry.object import PVector

start_time = time.time()

parser = argparse.ArgumentParser(description='Compute various NTK alignment quantities')

parser.add_argument('--task', required=True, type=str, help='Task',
                    choices=['mnist_fc', 'cifar10_vgg19', 'cifar10_resnet18'])
parser.add_argument('--depth', default=0, type=int, help='network depth (only works with MNIST MLP)')
parser.add_argument('--width', default=0, type=int, help='network width (MLP) or base for channels (VGG)')

parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='Momentum')

parser.add_argument('--diff', default=0., type=float, help='Proportion of difficult examples')
parser.add_argument('--diff-type', default='random', type=str, help='Type of difficult examples',
                    choices=['random', 'other'])

parser.add_argument('--align-train', action='store_true', help='Compute alignment with train set')
parser.add_argument('--align-test', action='store_true', help='Compute alignment with test set')
parser.add_argument('--align-easy-diff', action='store_true', help='Compute alignment with easy and difficult samples (requires diff > 0)')
parser.add_argument('--layer-align-train', action='store_true', help='Compute alignment with each layer separately (train set)')
parser.add_argument('--layer-align-test', action='store_true', help='Compute alignment with each layer separately (test set)')
parser.add_argument('--complexity', action='store_true', help='Compute trace(K) and norm(dw) in order to compute the complexity')

parser.add_argument('--no-centering', action='store_true', help='Disable centering when computing kernels')

parser.add_argument('--save-ntk-train', action='store_true', help='Save training set ntk')
parser.add_argument('--save-ntk-test', action='store_true', help='Save test set ntk')

parser.add_argument('--seed', default=1, type=int, help='Seed')
parser.add_argument('--epochs', default=100, type=int, help='epochs')

args = parser.parse_args()

model, dataloaders, criterion = get_task(args)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                      weight_decay=5e-4)

class RunningAverageEstimator:

    def __init__(self, gamma=.9):
        self.estimates = dict()
        self.gamma = gamma

    def update(self, key, val):
        if key in self.estimates.keys():
            self.estimates[key] = (self.gamma * self.estimates[key] +
                                   (1 - self.gamma) * val)
        else:
            self.estimates[key] = val

    def get(self, key):
        return self.estimates[key]

rae = RunningAverageEstimator()

def output_fn(x, t):
    return model(x)

def stopping_criterion(log):
    if (log.loc[len(log) - 1]['train_loss'] < 1e-2
            and log.loc[len(log) - 2]['train_loss'] < 1e-2):
        return True
    return False

def do_compute_ntk(iterations):
    return iterations == 0 or iterations in 5 * (1.15 ** np.arange(300)).astype('int')

# Training
def train(args, log, rae):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    iterations = 0

    if args.complexity:
        w_before = PVector.from_model(model).clone().detach()

    for epoch in range(args.epochs):
        print('\nEpoch: %d' % epoch)
        for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
            acc = pred.eq(targets.view_as(pred)).float().mean()

            rae.update('train_loss', loss.item())
            rae.update('train_acc', acc.item())

            if do_compute_ntk(iterations):
                to_log = pd.Series()
                to_log['time'] = time.time() - start_time
                if args.layer_align_train:
                    to_log['layer_align_train'] = \
                        layer_alignment(model, output_fn, dataloaders['micro_train'], 10,
                                        centering=not args.no_centering)
                if args.layer_align_test:
                    to_log['layer_align_test'] = \
                        layer_alignment(model, output_fn, dataloaders['micro_test'], 10,
                                        centering=not args.no_centering)
                if args.align_train or args.save_ntk_train:
                    to_log['align_train'], ntk = alignment(model, output_fn, dataloaders['micro_train'],
                                                           10, centering=not args.no_centering)
                    if args.save_ntk_train:
                        ntk_path = os.path.join(results_dir,'train_ntk_%.6d.pkl' % iterations)
                        torch.save(ntk, ntk_path)
                if args.align_test or args.save_ntk_test:
                    to_log['align_test'], ntk = alignment(model, output_fn, dataloaders['micro_test'],
                                                          10, centering=not args.no_centering)
                    if args.save_ntk_test:
                        ntk_path = os.path.join(results_dir,'test_ntk_%.6d.pkl' % iterations)
                        torch.save(ntk, ntk_path)
                if args.align_easy_diff:
                    to_log['align_easy_train'], ntk = alignment(model, output_fn,
                                                                dataloaders['micro_train_easy'],
                                                                10, centering=not args.no_centering)
                    to_log['align_diff_train'], ntk = alignment(model, output_fn,
                                                                dataloaders['micro_train_diff'],
                                                                10, centering=not args.no_centering)
                if args.complexity:
                    w_after = PVector.from_model(model).clone().detach()
                    to_log['norm_dw'] = torch.norm((w_after - w_before).get_flat_representation()).item()
                    w_before = w_after
                    to_log['trK'] = compute_trK(dataloaders['micro_train'], model, output_fn, 10)

                to_log['iteration'] = iterations
                to_log['epoch'] = epoch
                to_log['train_acc'], to_log['train_loss'] = rae.get('train_acc'), rae.get('train_loss')
                to_log['test_acc'], to_log['test_loss'] = test(dataloaders['mini_test'])


                log.loc[len(log)] = to_log
                print(log.loc[len(log) - 1])

                log.to_pickle(os.path.join(results_dir,'log.pkl'))

            iterations += 1

def test(loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    model.train()
    return correct / total, test_loss / (batch_idx + 1)


name = ''
for k, v in sorted(args.__dict__.items(), key=lambda a: a[0]):
    if (k not in ['save_ntk_train', 'save_ntk_test']
        and v is not False):
        name += '%s=%s,' % (k, str(v))
name = name[:-1]
results_dir = os.path.join('results', name)

try:
    os.mkdir(results_dir)
except:
    print('I will be overwriting a previous experiment')

columns = ['iteration', 'time', 'epoch',
           'train_loss', 'train_acc',
           'test_loss', 'test_acc']
if args.layer_align_train:
    columns.append('layer_align_train')
if args.layer_align_test:
    columns.append('layer_align_test')
if args.align_train or args.save_ntk_train:
    columns.append('align_train')
if args.align_test or args.save_ntk_test:
    columns.append('align_test')
if args.align_easy_diff:
    columns.append('align_easy_train')
    columns.append('align_diff_train')
if args.complexity:
    columns += ['trK', 'norm_dw']

log = pd.DataFrame(columns=columns)
train(args, log, rae)
