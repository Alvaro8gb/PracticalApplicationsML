import json

import pandas as pd

from globals import PATH_DS


# PACKING Y UNPACKING
def pack(data, target, features_names, class_names):
    return dict(data=data, target=target, features_names=features_names, class_names=class_names)


def unpack(ds):
    return ds["data"], ds["target"], ds["features_names"], ds["class_names"]


def load_ds(name: str):
    full = pd.read_pickle(PATH_DS + name + ".pkl")
    X, y, features_names, class_names = unpack(full)
    return X, y, features_names, class_names


def dump_ds(name: str, X, y, features_names, class_names):
    full = pack(X, y, features_names, class_names)
    path = PATH_DS + name + ".pkl"
    pd.to_pickle(full, path)
    print("Saved in ", path)


# HYPERS

def load_params(model: str) -> dict:
    with open(PATH_DS + "hipers.json") as f:
        params = json.load(f)

    return params[model]
