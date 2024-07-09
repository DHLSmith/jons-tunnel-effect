# Note taken from simplicity-bias...
# TODO: need to stop duplication and import this from a set of common utils!

from datetime import datetime
import random

import numpy as np
import pandas as pd
import torch
from torchbearer import Trial
from torchbearer.callbacks import TensorBoard, TensorBoardText, MultiStepLR, TorchScheduler
import torchbearer

FORCE_MPS = False


def set_seed(seed):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False


def get_device(device):
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps' if FORCE_MPS else 'cpu'
        else:
            device = 'cpu'

    return device


def fit_model(model, loss, opt, trainloader, valloader, epochs=1000, schedule=None, gamma=None, run_id=None, log_dir=None,
              model_file=None, resume=None, device='auto', verbose=0,  pre_extra_callbacks=None, extra_callbacks=None,
              acc='binary_acc', period=1):
    print('==> Setting up callbacks..')

    device = get_device(device)

    cb = []
    if log_dir is not None:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S') + "-run-" + str(run_id)
        tboard = TensorBoard(write_graph=False, comment=current_time, log_dir=log_dir)
        tboardtext = TensorBoardText(write_epoch_metrics=False, comment=current_time, log_dir=log_dir)

        @torchbearer.callbacks.on_start
        def write_params(_):
            params = {'model': str(model), 'loss': str(loss), 'opt': str(opt), 'trainloader': str(trainloader),
                      'valloader': str(valloader), 'schedule': str(schedule), 'run_id': str(run_id),
                      'log_dir': str(log_dir), 'model_file': str(model_file), 'resume': str(resume),
                      'device': str(device)}
            df = pd.DataFrame(params, index=[0]).transpose()
            tboardtext.get_writer(tboardtext.log_dir).add_text('params', df.to_html(), 1)

        cb.extend([tboard, tboardtext, write_params])

    if extra_callbacks is not None:
        if not isinstance(extra_callbacks, (list, tuple)):
            extra_callbacks = [extra_callbacks]
        cb.extend(extra_callbacks)

    if pre_extra_callbacks is not None:
        if not isinstance(pre_extra_callbacks, (list, tuple)):
            pre_extra_callbacks = [pre_extra_callbacks]
        cb = pre_extra_callbacks + cb

    if model_file is not None:
        cb.append(torchbearer.callbacks.MostRecent(model_file.replace(".pt", "_last.pt")))
        cb.append(torchbearer.callbacks.Interval(model_file, period=period, on_batch=False))
    if schedule is not None:
        cb.append(MultiStepLR(schedule, gamma=gamma))

    print('==> Training model..')
    print('using device: ' + device)
    metrics = ['loss', 'lr']
    if acc is not None:
        if not isinstance(acc, (list, tuple)):
            metrics.append(acc)
        else:
            metrics.extend(acc)
    trial = Trial(model, opt, loss, metrics=metrics, callbacks=cb)
    trial.with_generators(train_generator=trainloader,
                          val_generator=valloader).to(device)

    if resume is not None:
        print('resuming from: ' + resume)
        state = torch.load(resume)
        trial.load_state_dict(state)
        trial.replay()

    history = None
    if trainloader is not None:
        history = trial.run(epochs, verbose=verbose)
    metrics = trial.evaluate(data_key=torchbearer.TEST_DATA)

    return trial, history, metrics


def evaluate_model(model, test_loader, metrics, extra_callbacks=None, device='auto'):
    device = get_device(device)

    cb = []
    if extra_callbacks is not None:
        if not isinstance(extra_callbacks, (list, tuple)):
            extra_callbacks = [extra_callbacks]
        cb.extend(extra_callbacks)

    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]

    return (torchbearer.Trial(model, None, None, metrics=metrics, callbacks=cb, verbose=0)
            .with_val_generator(test_loader).to(device).evaluate())
