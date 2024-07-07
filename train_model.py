import argparse

import torchbearer
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, vgg19, vgg16
from torchvision.datasets import CIFAR10, CIFAR100

from utils.modelfitting import fit_model, set_seed
from utils.mlp import mlp6, mlp8, mlp10, mlp12


PARAMETERS = {
    'vgg': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'num_epochs': 160,
        'batch_size': 128,
        'milestones': [80, 120],
        'gamma': 0.1
    },
    'resnet': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'num_epochs': 164,
        'batch_size': 128,
        'milestones': [82, 123],
        'gamma': 0.1
    },
    'mlp': {
        'lr': 0.05,
        'momentum': 0.0,
        'weight_decay': 0,
        'num_epochs': 1000,
        'batch_size': 128,
        'milestones': None,
        'gamma': None
    }
}


def train(args):
    # No augmentation? interesting...
    if args.dataset == 'cifar10':
        train_set = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        val_set = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        n_classes = 10
    elif args.dataset == 'cifar100':
        train_set = CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
        val_set = CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
        n_classes = 100
    else:
        raise NotImplementedError

    if args.model.startswith('vgg'):
        params = PARAMETERS['vgg']
    elif args.model.startswith('resnet'):
        params = PARAMETERS['resnet']
    elif args.model.startswith('mlp'):
        params = PARAMETERS['mlp']
    else:
        raise NotImplementedError

    train_dl = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=2)

    model = globals()[args.model](num_classes=n_classes)
    loss = torch.nn.CrossEntropyLoss()  # presumably!

    optimiser = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'],
                                weight_decay=params['weight_decay'])

    if params['milestones'] is None:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=params['milestones'],
                                                         gamma=params['gamma'])

    fit_model(model, loss, optimiser, train_dl, val_dl, epochs=params['num_epochs'], schedule=scheduler,
              run_id=None, log_dir="./logs",
              model_file="./models/" + args.model + "-" + args.dataset + ".{epoch:03d}.pt",
              device='auto', verbose=1, acc='acc', period=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default='vgg19', choices=['vgg16', 'vgg19', 'resnet18',
                                                                       'resnet34', 'mlp6', 'mlp8', 'mlp10', 'mlp12'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])

    args = parser.parse_args()
    set_seed(42)

    train(args)
