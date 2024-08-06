import argparse
import copy
import gc
import os.path

import pandas as pd
import torch
import torchbearer
from torch import nn
from torch.utils.data import DataLoader
from torchbearer import Callback

from utils.analysis import AnalyserList, NameAnalyser, TrainableAnalyser
from utils.datasets import get_data
from utils.linear_probes import LinearProbe
from utils.modelfitting import evaluate_model, set_seed
from utils.models import parse_model_filename, get_model
from utils.rank import RankAnalyser, CovarianceSpectrumStatisticsAnalyser


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


class AnalysisHook(Callback):
    def __init__(self, analyser, layer_name):
        super().__init__()
        self.targets = None
        self.analyser = analyser
        self.layer_name = layer_name

    def on_sample(self, state):
        self.targets = state[torchbearer.Y_TRUE]

    def on_sample_validation(self, state):
        self.targets = state[torchbearer.Y_TRUE]

    def __call__(self, m, input, output):
        self.analyser(output, self.targets, m, self.layer_name)


def install_hooks(mdl, train_set):
    analysers = {}
    handles = []
    callbacks = []
    for name, m in mdl.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            lp = LinearProbe()

            print(f"training probe for {name}")
            fe = FeatureExtractor(mdl, name)
            lp.train(train_set, fe)
            analysers[name] = AnalyserList(NameAnalyser(), CovarianceSpectrumStatisticsAnalyser(), lp)
            del fe

            cb = AnalysisHook(analysers[name], name)
            handles.append(m.register_forward_hook(cb))
            callbacks.append(cb)

    return analysers, handles, callbacks


def perform_analysis(analysers, params=None):
    results = []

    for name, analyser in analysers.items():
        rec = analyser.get_result()

        if params is not None:
            rec.update(params)

        results.append(rec)

    return pd.DataFrame.from_records(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=str, default='/Users/jsh2/data')
    parser.add_argument('--output', type=str, default='./results/')
    parser.add_argument('-nf', '--num-features', type=int, default=8000)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--skip', default=False, action='store_true')
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()

    for filename in args.files:
        params = parse_model_filename(filename)
        set_seed(params['seed'])
        out_filename = filename.split('/')[-1].replace('.pt', '-train.csv' if args.train else '-test.csv')
        if args.skip and os.path.exists(f"{args.output}/{out_filename}"):
            continue

        print(f"{args.output}/{out_filename}")

        num_classes, train_set, val_set = get_data(params['dataset'], args.root)
        if args.train:
            dl = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=0)
        else:
            dl = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

        mdl = get_model(params['model'])(num_classes=num_classes)
        if os.path.exists(filename):
            state = torch.load(filename, map_location=torch.device("cpu"))
            mdl.load_state_dict(state["model"], assign=True)
        else:
            print("Warning: weights file didn't exist. Going to log a random model")

        with torch.no_grad():
            analysers, handles, callbacks = install_hooks(mdl, train_set)

            metrics = evaluate_model(mdl, dl, 'acc', verbose=2, extra_callbacks=callbacks)

            for h in handles:
                h.remove()

            params.update(metrics)

            df = perform_analysis(analysers, params)
            df.to_csv(f"{args.output}/{out_filename}")

        del num_classes, train_set, val_set, dl, mdl, analysers, metrics, params
        gc.collect()


if __name__ == '__main__':
    main()
