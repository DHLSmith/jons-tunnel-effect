import argparse

import torch
from torch.utils.data import DataLoader

from utils.datasets import get_data
from utils.modelfitting import fit_model, set_seed
from utils.models import get_model, get_model_filename

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


def get_parameters(args):
    if args.model.startswith('vgg'):
        params = PARAMETERS['vgg']
    elif args.model.startswith('resnet'):
        params = PARAMETERS['resnet']
    elif args.model.startswith('mlp'):
        params = PARAMETERS['mlp']
    else:
        raise NotImplementedError
    if args.lr is None:
        args.lr = params['lr']
    params.update(vars(args))
    return params


def train(args):
    n_classes, train_set, val_set = get_data(args.dataset, args.root)

    params = get_parameters(args)

    train_dl = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=3)
    val_dl = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=1)

    model = get_model(args.model)(num_classes=n_classes)
    loss = torch.nn.CrossEntropyLoss()  # presumably!

    optimiser = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'],
                                weight_decay=params['weight_decay'])

    fit_model(model, loss, optimiser, train_dl, val_dl, epochs=params['num_epochs'], schedule=params['milestones'],
              gamma=params['gamma'], run_id=f"{args.model}-{args.dataset}", log_dir=f"{args.output}/logs",
              model_file=get_model_filename(params),
              device='auto', verbose=2, acc='acc', period=10, args=params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default='vgg19', choices=['vgg16', 'vgg19', 'vgg16_bn',
                                                                       'vgg19_bn', 'resnet18', 'resnet34', 'mlp6',
                                                                       'mlp8', 'mlp10', 'mlp12'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--root', type=str, default='/scratch/jsh2/datasets')
    parser.add_argument('--output', type=str, default='/scratch/jsh2/tunnel')

    args = parser.parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()

