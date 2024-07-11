import timeit

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16, vgg19, resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.modelfitting import get_device

mdl = vgg19(num_classes=10)
# mdl = resnet18(num_classes=10)
state = torch.load("models/vgg19-cifar10.159_last.pt", map_location=torch.device("cpu"))
mdl.load_state_dict(state["model"], assign=True)

# with torch.no_grad():
# for name, m in mdl.named_modules():
#     if "conv" in name or isinstance(m, nn.Conv2d):
#         print(name)
#         weights = m.weight.view(m.weight.shape[0], -1)
#         u, s, v = torch.svd(weights, compute_uv=False)
#         scount = (s > 1e-3 * s.max()).sum()
#         print(scount, weights.shape, s.max(), s.min(), s)


tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_set = CIFAR10(root='/ssd/data', train=True, download=True, transform=tf)
# val_set = CIFAR10(root='/ssd/data', train=False, download=True, transform=tf)
train_set = CIFAR10(root='/Users/jsh2/data', train=True, download=True, transform=tf)
val_set = CIFAR10(root='/Users/jsh2/data', train=False, download=True, transform=tf)
n_classes = 10

train_dl = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
val_dl = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

features = {}


def make_hook(name):
    def hook(m, input, output):
        features[name].append(output)

    return hook


def estimate_rank(features, n=8000, mode='aTa', thresh=1e-3):
    """
    Estimate the rank of mean-centred features matrix (equivalent to rank of covariance matrix of either features or samples)
    :param features: the features, one example per row
    :param n: the amount to sample in order to reduce computation time. -1 to disable sampling.
    :param mode: 'aTa' to use features covariance; 'aaT' to use examples x examples
    :param thresh: threshold as a percentage of largest s.v. to use to estimate the rank
    :return: the estimated rank
    """
    if mode == 'aTa':
        if n > 0:
            perm = torch.randperm(features.shape[1])
            idx = perm[:n]
            f = features[:, idx]
        else:
            f = features

        # cov = (f - f.mean(dim=0)).T @ (f - f.mean(dim=0))
        cov = torch.cov(f.T)
        # s = torch.linalg.svdvals(cov)
        # count = (s > thresh * s.max()).sum()
        # return count.cpu().item()
        return torch.linalg.matrix_rank(cov, hermitian=True, rtol=thresh)
    elif mode == 'aaT':
        if n > 0:
            perm = torch.randperm(features.shape[0])
            idx = perm[:n]
            f = features[idx, :]
        else:
            f = features

        cov = (f - f.mean(dim=0)) @ (f - f.mean(dim=0)).T
        s = torch.linalg.svdvals(cov)
        count = (s > thresh * s.max()).sum()
        return count.cpu().item()


with torch.no_grad():
    for name, m in mdl.named_modules():
        if "conv" in name or isinstance(m, nn.Conv2d):
            features[name] = []

            m.register_forward_hook(make_hook(name))

    mdl.to(get_device("auto"))
    for e, (x, y) in tqdm(enumerate(train_dl)):
        mdl(x.to(get_device("auto")))
        if e == 100:
            break

    for name, fvs in features.items():
        f = torch.cat(fvs, dim=0)
        f = f.view(f.shape[0], -1)
        # f = torch.relu(f)
        rank = estimate_rank(f, n=8000, thresh=1e-3)
        print(name, rank)
