import abc
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
            result.update(analyser.get_result())

        return result


class TrainableAnalyser(Analyser):
    @abc.abstractmethod
    def train(self, dataset: Dataset, feature_extractor: Callable):
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

