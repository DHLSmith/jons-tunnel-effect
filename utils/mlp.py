from torch import nn


def _make_mlp(num_classes, depth):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=32 * 32 * 3, out_features=1024),
        nn.ReLU(),
        *[nn.Linear(in_features=1024, out_features=1024), nn.ReLU()] * depth - 2,
        nn.Linear(in_features=1024, out_features=num_classes)
    )


def mlp6(num_classes):
    return _make_mlp(num_classes, 6)


def mlp8(num_classes):
    return _make_mlp(num_classes, 8)


def mlp10(num_classes):
    return _make_mlp(num_classes, 10)


def mlp12(num_classes):
    return _make_mlp(num_classes, 12)