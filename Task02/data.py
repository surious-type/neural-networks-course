import hashlib
import json
from pathlib import Path

import torch


# Data Generation
def get_data(seed=42):
    generator = torch.Generator().manual_seed(seed)

    n_samples = 10000
    n_train = 8000

    x = (1e-15 + (1 - 1e-15) * torch.rand(n_samples, generator=generator)).unsqueeze(1)
    y = x**2

    idx = torch.randperm(n_samples, generator=generator)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return x_train, y_train, x_val, y_val


def make_config(config):
    layers, loss, func, optimizer_class, epochs, batch_size = config

    if isinstance(layers, tuple):
        model_params = {
            "input_size": 1,
            "one": layers[0],
            "two": layers[1],
            "out": 1,
        }
    else:
        model_params = {
            "input_size": 1,
            "one": layers,
            "out": 1,
        }

    return {
        "model_params": str(model_params),
        "model_loss": loss.__class__.__name__,
        "func": func.__name__,
        "optimizer_class": optimizer_class,
        "epochs": epochs,
        "batch_size": batch_size,
    }


class Store:
    CONFIG = [
        "model_params",
        "model_loss",
        "func",
        "optimizer_class",
        "epochs",
        "batch_size",
    ]

    METRICS = [
        "loss",
        "mape",
    ]

    FIELDS = [
        "run_id",
        *CONFIG,
        *METRICS,
    ]

    def __init__(self, filename):
        self.filename = filename
        self.path = Path(filename)
        self.ids = set()

        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(";".join(self.FIELDS) + "\n")
        else:
            with open(self.filename, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    values = line.strip().split(";")
                    record = dict(zip(self.FIELDS, values))
                    self.ids.add(record["run_id"])

    @classmethod
    def get_id(cls, run_config):
        config = {key: str(run_config[key]) for key in cls.CONFIG}

        serialized = json.dumps(
            config,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]

    def exists(self, run_config):
        return self.get_id(run_config) in self.ids

    def save(self, config, metrics):
        run_id = self.get_id(config)
        record = {
            "run_id": run_id,
            **config,
            **metrics,
        }

        with open(self.filename, "a") as f:
            f.write(";".join(str(v) for v in record.values()) + "\n")

        self.ids.add(run_id)
