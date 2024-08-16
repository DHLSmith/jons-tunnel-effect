from types import MappingProxyType

import torch
import torchvision.transforms.functional
from torch import nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from utils.analysis import Analyser, TrainableAnalyser
from utils.modelfitting import fit_model, evaluate_model, get_device

DEFAULT_LINEAR_PROBE_OPTIM_PARAMS = MappingProxyType({'lr': 0.001, 'weight_decay': 0})


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_name):
        super().__init__()
        self.model = model
        self.features = None

        def hook(_, __, output):
            self.features = output
            raise StopIteration

        for name, module in model.named_modules():
            if layer_name == name:
                self.hndl = module.register_forward_hook(hook)
                break

    def __del__(self):
        self.unhook()

    def unhook(self):
        if self.hndl is not None:
            self.hndl.remove()
            self.hndl = None

    def forward(self, x):
        try:
            self.model.eval()
            self.model(x)
        except StopIteration:
            pass
        return self.features

    def create_tensor_dataset(self, dataset: Dataset, batch_size: int = 128):
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        features = []
        classes = []

        for x, y in tqdm(loader):
            with torch.no_grad():
                features.append(self(x))
                classes.append(y)

        features = torch.cat(features)

        return TensorDataset(features, torch.cat(classes))

    # def create_dynamic_dataset(self, dataset: Dataset, batch_size: int = 128):
    #     def get_feature(x):
    #         with torch.no_grad():
    #             return self(x.unsqueeze)[0]
    #     class DynDataset(Dataset):
    #         def __init__(self, ds: Dataset):
    #             self.dataset = ds
    #
    #         def __getitem__(self, index):
    #             x, y = self.dataset[index]
    #
    #             f = get_feature(x)
    #
    #             return f, y
    #
    #         def __len__(self):
    #             return len(self.dataset)
    #
    #     return DynDataset(dataset)


class LinearProbe(TrainableAnalyser):
    def __init__(self, num_classes=10, batch_size=512, num_epochs=30, optimizer=Adam,
                 optimizer_params=DEFAULT_LINEAR_PROBE_OPTIM_PARAMS, device='auto'):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.model = None
        self.device = get_device(device)
        self.predictions = []
        self.train_metrics = None
        self.mean = None
        self.std = None

    def train(self, dataset: TensorDataset):
        if len(dataset.tensors[0].shape) == 4:
            self.mean = dataset.tensors[0].mean(dim=(0, 2, 3)).tolist()
            self.std = dataset.tensors[0].mean(dim=(0, 2, 3)).tolist()
            ndata = torchvision.transforms.functional.normalize(dataset.tensors[0], self.mean, self.std)
            ndata = ndata.view(dataset.tensors[0].shape[0], -1)
        else:
            self.mean = dataset.tensors[0].mean(dim=0)
            self.std = dataset.tensors[0].mean(dim=0)
            ndata = (dataset.tensors[0] - self.mean) / self.std

        dataset = TensorDataset(ndata, dataset.tensors[1])

        self.model = nn.Linear(dataset[0][0].shape[0], self.num_classes)
        loss = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_params)

        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        trial, history, _ = fit_model(self.model, loss, optimizer, loader, None, epochs=self.num_epochs,
                                      device=self.device, verbose=1)
        self.train_metrics = {'train_acc': history[-1]['acc']}

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name) -> None:
        features = features.to(self.model.weight.device)

        if len(features.shape) == 4:
            features = torchvision.transforms.functional.normalize(features, self.mean, self.std)
            features = features.view(features.shape[0], -1)
        else:
            features = (features - self.mean.to(self.model.weight.device)) / self.std.to(self.model.weight.device)

        pred = self.model(features)
        self.predictions.append((pred.cpu(), classes.cpu()))

    def get_result(self) -> dict:
        res = {}
        res.update(evaluate_model(nn.Identity(), self.predictions, 'acc'))
        res.update(self.train_metrics)
        return res

