import argparse
import gc
import os.path

import pandas as pd
import torch
from torch import nn
from torch.linalg import LinAlgError
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datasets import get_data
from utils.modelfitting import evaluate_model, set_seed
from utils.models import parse_model_filename, get_model
from utils.rank import estimate_rank


def make_hook(layer_name, features):
    def hook(m, input, output):
        features[layer_name].append(output.to("cpu"))

    return hook


def install_hooks(mdl):
    layers = {}
    features = {}
    handles = []
    for name, m in mdl.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            features[name] = []

            handles.append(m.register_forward_hook(make_hook(name, features)))
            layers[name] = m

    return layers, features


def perform_analysis(features, classes, layers, params=None, n=8000):
    results = []

    try:
        for name, fvs in tqdm(features.items()):
            rec = {'name': name}
            if params is not None:
                rec.update(params)

            f = torch.cat(fvs, dim=0)
            f = f.view(f.shape[0], -1)
            rank = estimate_rank(f, n=n, thresh=1e-3)

            w = layers[name].weight
            w = w.view(w.shape[0], -1)
            w_rank = torch.linalg.matrix_rank(w, hermitian=False, rtol=1e-3).cpu().item()

            rec['features_rank'] = rank
            rec['features_dim'] = f.shape[1]
            rec['normalized_features_rank'] = rank / min(f.shape[1], f.shape[0])
            rec['weights_rank'] = w_rank

            for c in range(classes.max() + 1):
                cf = f[classes == c]
                cr = estimate_rank(cf, n=n, thresh=1e-3)
                rec['features_rank_'+str(c)] = cr
                rec['normalized_features_rank_'+str(c)] = cr / min(cf.shape[1], cf.shape[0])

            results.append(rec)
    except LinAlgError:
        pass

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
            layers, features = install_hooks(mdl)

            metrics = evaluate_model(mdl, dl, 'acc', verbose=2)
            params.update(metrics)

            classes = torch.cat([y.cpu() for _, y in dl])

            df = perform_analysis(features, classes, layers, params, n=args.num_features)
            df.to_csv(f"{args.output}/{out_filename}")

        del num_classes, train_set, val_set, dl, mdl, layers, features, metrics, params
        gc.collect()


if __name__ == '__main__':
    main()

