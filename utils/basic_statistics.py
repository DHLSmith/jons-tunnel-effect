import torch
from torch import nn

from utils.analysis import Analyser
from utils.running_stats import Variance


class FeatureStats(Analyser):
    def __init__(self):
        self.channel_sparsity = Variance()
        self.channel_activations = Variance()
        self.feature_activations = Variance()

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        channel_features = features.view(features.shape[0], 1, -1)

        self.channel_sparsity.add(channel_features.count_nonzero(dim=-1) / channel_features.shape[-1])
        self.channel_activations.add(channel_features.mean(dim=-1))
        self.feature_activations.add(features)

    def get_result(self) -> dict:
        rec = dict()
        rec['channel_sparsity'] = self.channel_sparsity
        rec['channel_activations'] = self.channel_activations
        rec['feature_activations'] = self.feature_activations
        return rec


