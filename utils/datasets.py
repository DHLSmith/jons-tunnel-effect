from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


def get_data(dataset, root="/ssd"):
    # No augmentation? interesting...
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if dataset == 'cifar10':
        train_set = CIFAR10(root=root, train=True, download=True, transform=tf)
        val_set = CIFAR10(root=root, train=False, download=True, transform=tf)
        n_classes = 10
    elif dataset == 'cifar100':
        train_set = CIFAR100(root=root, train=True, download=True, transform=tf)
        val_set = CIFAR100(root=root, train=False, download=True, transform=tf)
        n_classes = 100
    else:
        raise NotImplementedError

    return n_classes, train_set, val_set

