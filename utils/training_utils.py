import logging
import random
import numpy as np
import torch
from torch import optim


def set_logger(log_path, log_level=logging.INFO, logger_name="log"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_optimizer(params, lr, betas, eps, momentum, optimizer_name):
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=lr, betas=betas, eps=eps)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(params, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError("optimizer name is not correct")
    return optimizer


def split_parameters_for_adamw(named_parameters):
    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    return gain_or_bias_params, rest_params


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster
