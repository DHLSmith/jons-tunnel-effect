import torch
from torch import nn as nn
from torch.linalg import LinAlgError

from utils.analysis import Analyser
from utils.running_stats import Covariance


def estimate_rank(features: torch.Tensor, n: int = 8000, mode: str = 'aTa', threshold: float = 1e-3) -> int:
    """
    Estimate the rank of mean-centred features matrix (equivalent to rank of covariance matrix of either features or samples)
    :param features: the features, one example per row
    :param n: the amount to sample in order to reduce computation time. -1 to disable sampling.
    :param mode: 'aTa' to use features covariance; 'aaT' to use examples x examples; 'a' to use mean-centred features matrix
    :param threshold: threshold as a percentage of largest s.v. to use to estimate the rank
    :return: the estimated rank
    """
    if mode == 'aTa':
        if n > 0:
            perm = torch.randperm(features.shape[1])
            idx = perm[:n]
            f = features[:, idx]
        else:
            f = features

        cov = torch.cov(f.T)
        return torch.linalg.matrix_rank(cov, hermitian=True, rtol=threshold).cpu().item()
    elif mode == 'aaT':
        if n > 0:
            perm = torch.randperm(features.shape[0])
            idx = perm[:n]
            f = features[idx, :]
        else:
            f = features

        cov = (f - f.mean(dim=0)) @ (f - f.mean(dim=0)).T
        return torch.linalg.matrix_rank(cov, hermitian=True, rtol=threshold).cpu().item()
    elif mode == 'a':
        if n > 0:
            perm = torch.randperm(features.shape[0])
            idx = perm[:n]
            f = features[idx, :]
        else:
            f = features

        s = torch.linalg.svdvals(f - f.mean(dim=0)) ** 2
        return (s > (s.max() * threshold)).sum().cpu().item()


def compute_cov_spectrum_stats(covariance: torch.Tensor, threshold=1e-3, taps=10) -> dict:
    s = torch.linalg.svdvals(covariance)

    stats = dict()
    stats['mean'] = s.mean().cpu().item()
    stats['max'] = s[0].item()
    stats['features_rank'] = (s > (s.max() * threshold)).sum().cpu().item()
    stats['features_rank_val'] = s[stats['features_rank'] - 1].item()
    stats['half_rank_val'] = s[(stats['features_rank'] - 1) // 2].item()
    stats['quarter_rank_val'] = s[(stats['features_rank'] - 1) // 4].item()

    spectrum = torch.nn.functional.interpolate(s.view(1, 1, -1), size=taps, mode='nearest')[0, 0].cpu()
    for i in range(taps):
        stats[f'normalised_spectrum_{i}'] = spectrum[i].item()

    return stats


class RankAnalyser(Analyser):
    def __init__(self, mode='aTa', n=8000, threshold=1e-3):
        self.n = n
        self.indices = None
        self.features_dim = None
        self.mode = mode
        self.threshold = threshold
        self.covar = Covariance()

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        if self.indices is None:
            self.features_dim = features.shape[1]
            self.indices = torch.randperm(features.shape[1])[:self.n]

        f = features.view(features.shape[0], -1)[:, self.indices]
        self.covar.add(f)

    def get_result(self) -> dict:
        try:
            covar = self.covar.covariance()
            rank = torch.linalg.matrix_rank(covar, hermitian=True, rtol=self.threshold).cpu().item()

            rec = dict()
            rec['features_rank'] = rank
            rec['features_dim'] = self.features_dim
            rec['normalized_features_rank'] = rank / min(self.features_dim, covar.shape[0])
            return rec
        except LinAlgError:
            return {}


class PerClassRankAnalyser(RankAnalyser):
    def __init__(self, mode='aTa', n=8000, threshold=1e-3):
        super().__init__(mode, n, threshold)

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        if self.indices is None:
            self.features_dim = features.shape[1]
            self.indices = torch.randperm(features.shape[1])[:self.n]
            self.covar = {}

        f = features.view(features.shape[0], -1)[:, self.indices]

        for c in classes.unique():
            if c not in self.covar:
                self.covar[c] = Covariance()

            cf = f[classes == c]
            self.covar[c].add(cf)

    def get_result(self) -> dict:
        rec = dict()
        for c in self.covar.keys():
            try:
                covar = self.covar[c].covariance()
                cr = torch.linalg.matrix_rank(covar, hermitian=True, rtol=self.threshold).cpu().item()

                rec[f'features_rank_{c}'] = cr
                rec[f'normalized_features_rank_{c}'] = cr / min(self.features_dim, self.n)
            except LinAlgError:
                pass

        return rec


class LayerWeightRankAnalyser(RankAnalyser):
    def __init__(self, mode='aTa', n=8000, threshold=1e-3):
        super().__init__(mode, n, threshold)
        self.w_rank = None

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        if layer is not None and self.w_rank is None:
            w = layer.weight
            w = w.view(w.shape[0], -1)
            self.w_rank = torch.linalg.matrix_rank(w, hermitian=False, rtol=self.threshold).cpu().item()

    def get_result(self) -> dict:
        return {'weights_rank': self.w_rank}


class CovarianceSpectrumStatisticsAnalyser(RankAnalyser):
    def __init__(self, n=8000, threshold=1e-3, taps=10):
        super().__init__(n=n, threshold=threshold)
        self.taps = taps

    def get_result(self) -> dict:
        stats = compute_cov_spectrum_stats(self.covar.covariance(), threshold=self.threshold, taps=self.taps)
        stats['features_dim'] = self.features_dim
        stats['normalized_features_rank'] = stats['features_rank'] / min(self.features_dim, self.n)
        return stats
