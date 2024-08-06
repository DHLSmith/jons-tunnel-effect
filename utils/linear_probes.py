from types import MappingProxyType
from typing import Callable

import torch
from torch import nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

        for name, module in model.named_modules():
            if layer_name == name:
                self.hndl = module.register_forward_hook(hook)

    def __del__(self):
        self.hndl.remove()

    def forward(self, x):
        self.model.eval()
        self.model(x)
        return self.features.view(self.features.shape[0], -1)


class LinearProbe(TrainableAnalyser):
    def __init__(self, num_classes=10, batch_size=512, num_epochs=1, optimizer=Adam,
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

    def train(self, dataset: Dataset, feature_extractor: FeatureExtractor):
        class TrModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.model = nn.LazyLinear(num_classes)

            def forward(self, x):
                with torch.no_grad():
                    x = feature_extractor(x)
                return self.model(x)

        feature_extractor = feature_extractor.to(device=self.device)
        trmodel = TrModel(self.num_classes).to(self.device)
        self.model = trmodel.model

        loss = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_params)

        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        trial, history, _ = fit_model(trmodel, loss, optimizer, loader, None, epochs=self.num_epochs,
                                      device=self.device, verbose=2)
        self.train_metrics = history[-1]

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name) -> None:
        pred = self.model(features.to(self.model.weight.device).view(features.shape[0], -1))
        self.predictions.append((pred.cpu(), classes.cpu()))

    def get_result(self) -> dict:
        res = evaluate_model(nn.Identity(), self.predictions, 'acc')
        res.update(self.train_metrics)
        return res

