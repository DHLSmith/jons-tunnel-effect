import torch


def estimate_rank(features, n=8000, mode='aTa', thresh=1e-3):
    """
    Estimate the rank of mean-centred features matrix (equivalent to rank of covariance matrix of either features or samples)
    :param features: the features, one example per row
    :param n: the amount to sample in order to reduce computation time. -1 to disable sampling.
    :param mode: 'aTa' to use features covariance; 'aaT' to use examples x examples; 'a' to use mean-centred features matrix
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
        return torch.linalg.matrix_rank(cov, hermitian=True, rtol=thresh).cpu().item()
    elif mode == 'aaT':
        if n > 0:
            perm = torch.randperm(features.shape[0])
            idx = perm[:n]
            f = features[idx, :]
        else:
            f = features

        cov = (f - f.mean(dim=0)) @ (f - f.mean(dim=0)).T
        return torch.linalg.matrix_rank(cov, hermitian=True, rtol=thresh).cpu().item()
    elif mode == 'a':
        if n > 0:
            perm = torch.randperm(features.shape[0])
            idx = perm[:n]
            f = features[idx, :]
        else:
            f = features

        s = torch.linalg.svdvals(f - f.mean(dim=0)) ** 2
        return (s > (s.max() * thresh)).sum().cpu().item()
