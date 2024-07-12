import timeit

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16, vgg19, resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from utils import modelfitting
from utils.modelfitting import get_device
from utils.rank import estimate_rank

# mdl = vgg19(num_classes=10)
mdl = resnet18(num_classes=10)
# state = torch.load("models/vgg19-cifar10.159_last.pt", map_location=torch.device("cpu"))
state = torch.load("models/resnet18-cifar10.163_last.pt", map_location=torch.device("cpu"))
mdl.load_state_dict(state["model"], assign=True)

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
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
        features[name].append(output.to("cpu"))

    return hook


with torch.no_grad():
    layers = {}
    for name, m in mdl.named_modules():
        if "conv" in name or isinstance(m, nn.Conv2d):
            features[name] = []

            m.register_forward_hook(make_hook(name))
            layers[name] = m

    mdl.to(get_device("auto"))
    for e, (x, y) in tqdm(enumerate(train_dl)):
        mdl(x.to(get_device("auto")))
        if e == 100:
            break

    for name, fvs in features.items():
        f = torch.cat(fvs, dim=0)
        f = f.view(f.shape[0], -1)
        rank = estimate_rank(f, n=4000, thresh=1e-3)

        w = layers[name].weight
        w = w.view(w.shape[0], -1)
        w = torch.cat([w, w], dim=0)
        w_rank = torch.linalg.matrix_rank(w, hermitian=False, rtol=1e-3).cpu().item()
        print(name, rank, f.shape[1], rank/f.shape[1], w_rank)
