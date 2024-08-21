import abc
import copy
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Analyser(abc.ABC):
    def __call__(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        self.process_batch(features, classes, layer, name)

    @abc.abstractmethod
    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        pass

    @abc.abstractmethod
    def get_result(self) -> dict:
        pass


class AnalyserList(Analyser):
    def __init__(self, *args: Analyser):
        super().__init__()
        self.analysers = args

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str):
        for analyser in self.analysers:
            analyser(features, classes, layer, name)

    def get_result(self) -> dict:
        result = dict()

        for analyser in self.analysers:
            clz = type(analyser).__name__

            if clz == 'PerClassAnalyser' and hasattr(analyser, 'analyser'):
                clz = 'PerClass' + type(analyser.analyser).__name__

            for k, v in analyser.get_result().items():
                result[f"{clz}.{k}"] = v

        return result


class PerClassAnalyser(Analyser):
    def __init__(self, analyser):
        super().__init__()
        self.analyser = analyser
        self.analysers = {}

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str):
        for c in classes.unique():
            if c not in self.analysers:
                self.analysers[c] = copy.deepcopy(self.analyser)

            cf = features[classes == c]
            self.analysers[c].add(cf)

    def get_result(self) -> dict:
        result = dict()

        for c in self.analysers.keys():
            r = self.analysers[c].get_result()
            for k, v in r.items():
                result[f"{k}_{c}"] = v

        return result


class PerClassVersusAnalyser(PerClassAnalyser):
    def __init__(self, analyser):
        super().__init__(analyser)

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str):
        for c in classes.unique():
            if c not in self.analysers:
                self.analysers[c] = copy.deepcopy(self.analyser)
                self.analysers[f"~{c}"] = copy.deepcopy(self.analyser)

            cf = features[classes == c]
            self.analysers[c].add(cf)

            ncf = features[classes != c]
            self.analysers[f"~{c}"].add(ncf)

    def get_result(self) -> dict:
        result = dict()

        for c in self.analysers.keys():
            r = self.analysers[c].get_result()
            for k, v in r.items():
                result[f"{k}_{c}"] = v

        return result


class TrainableAnalyser(Analyser):
    @abc.abstractmethod
    def train(self, dataset: Dataset):
        pass


class NameAnalyser(Analyser):
    """
    Just logs the layer name
    """

    def __init__(self):
        super().__init__()
        self.name = None

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        if self.name is None:
            self.name = name

    def get_result(self) -> dict:
        return {'name': self.name}
