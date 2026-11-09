import math
from itertools import product
from multiprocessing import Pool

import torch
from data import Store, get_data, make_config
from models import MLP, MLP3
from torch import nn, optim
from train import run

lr = 0.001
optimizers = {
    # "Momentum": lambda params: optim.SGD(params, lr=lr, momentum=0.9),
    # "Nesterov": lambda params: optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
    # "Adagrad": lambda params: optim.Adagrad(params, lr=lr),
    "RMSprop": lambda params: optim.RMSprop(params, lr=lr),
    # "Adadelta": lambda params: optim.Adadelta(params, lr=lr),
    "Adam": lambda params: optim.Adam(params, lr=lr),
}


def relax(config):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    layers, loss, func, optimizer_class, epochs, batch_size = config
    if isinstance(layers, tuple):
        model_params = {"input_size": 1, "one": layers[0], "two": layers[1], "out": 1}
        model = MLP3(**model_params, func=func)
    else:
        model_params = {"input_size": 1, "one": layers, "out": 1}
        model = MLP(**model_params, func=func)
    optimizer = optimizers[optimizer_class](model.parameters())
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: math.exp(-0.01 * (epoch % epochs)),
    )
    x_train, y_train, x_val, y_val = get_data()
    _, test_errors, test_mapes = run(
        model,
        loss,
        optimizer,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs,
        sampling=True,
        batch_size=batch_size,
        scheduler=scheduler,
    )
    return make_config(config), {"loss": test_errors[-1][1], "mape": test_mapes[-1][1]}


if __name__ == "__main__":
    store = Store("results_MLP3.csv")

    epochs = [200]
    batch_sizes = [
        32,
        64,
        128,
        # 256,
        # 512,
    ]

    funcs = [
        # nn.ReLU,
        # nn.Sigmoid,
        # nn.Tanh,
        # nn.ELU,
        nn.LeakyReLU,
        nn.SiLU,
        nn.Mish,
    ]
    model_losses = [nn.MSELoss(), nn.L1Loss()]

    # configurations = product(
    #     range(10, 200, 10),
    #     model_losses,
    #     funcs,
    #     optimizers.keys(),
    #     epochs,
    #     batch_sizes,
    # )
    configurations = product(
        ((i, j) for i in range(10, 190, 10) for j in range(10, 200 - i, 10)),
        model_losses,
        funcs,
        optimizers.keys(),
        epochs,
        batch_sizes,
    )

    configurations = (
        config for config in configurations if not store.exists(make_config(config))
    )

    with Pool(processes=8) as pool:
        for config, metrics in pool.imap_unordered(relax, configurations, chunksize=1):
            store.save(config, metrics)
            print(
                f"Model: {config['model_params']}, Func: {config['func']}, Optimizer: {config['optimizer_class']}, Batch size: {config['batch_size']}, {config['model_loss']}: {metrics['loss']}, MAPE: {metrics['mape']}"
            )
