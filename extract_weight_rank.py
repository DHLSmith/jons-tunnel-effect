import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16
from torchvision.transforms import transforms

mdl = vgg16(num_classes=10)
state = torch.load("models/vgg16-cifar10.159_last.pt", map_location=torch.device("cpu"))
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


with torch.no_grad():
    for name, m in mdl.named_modules():
        if "conv" in name or isinstance(m, nn.Conv2d):
            features[name] = []

            m.register_forward_hook(make_hook(name))

    for x, y in val_dl:
        mdl(x)

    for name, fvs in features.items():
        f = torch.cat(fvs, dim=0)
        f = f.view(f.shape[0], -1)

        s = torch.linalg.svdvals(f - f.mean(dim=0)) ** 2
        scount = (s > 1e-3 * s.max()).sum()
        print("svd", name, scount)

        perm = torch.randperm(f.shape[1])
        idx = perm[:8000]
        f = f[:, idx]

        cov = (f - f.mean(dim=0)).T @ (f - f.mean(dim=0))
        s = torch.linalg.svdvals(cov)
        scount = (s > 1e-3 * s.max()).sum()
        print(name, scount)

        print()
