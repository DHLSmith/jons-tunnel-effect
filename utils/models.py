import importlib
from typing import Union

from torch import nn


def get_model(name: str) -> Union[nn.Module, None]:
    modules = [
        importlib.import_module("torchvision.models"),
        importlib.import_module("utils.mlp")
    ]
    for module in modules:
        try:
            return getattr(module, name)
        except AttributeError:
            pass

    return None


def get_model_filename(params: dict) -> str:
    fn = "{params['model']}-{params['dataset']}-lr{params['lr']}-seed{params['seed']}"

    return f"{params['output']}/models/{fn}" + "-{epoch:03d}.pt"


def parse_model_filename(filename: str) -> dict:
    params = {}

    dir_parts = filename.split("/")
    if len(dir_parts) > 1:
        params["output"] = "/".join(dir_parts[0:-1])

    fn_parts = dir_parts[-1].split("-")
    params["model"] = fn_parts[0]
    params["dataset"] = fn_parts[1]
    params["lr"] = float(fn_parts[2].replace("lr", ""))
    params["seed"] = int(fn_parts[3].replace("seed", ""))
    params["epoch"] = int(fn_parts[4].replace(".pt", "").replace("_last", ""))

    return params
