import logging
import os
import time
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from utils.training_utils import get_optimizer, cosine_lr, split_parameters_for_adamw


class PLWrapper(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, src):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        # self.global_step: the number of training steps completed from the beginning of training
        # batch_idx: current step index inside a single epoch
        # self.trainer.num_training_batches: total number of batches in the training set for one epoch
        # progress bar display:
        # Epoch 2:  10%|█▏          | 5/50,
        # in 5/50, the "50" contains train step + test step if test is enabled in that epoch.

        # step the lr scheduler
        self.scheduler(self.global_step)
        # log the lr
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("my_lr", current_lr, prog_bar=True, on_step=True)

        src, pos, neg = batch["input"], batch["pos"], batch["neg"]
        src_emb = self.model(src)
        pos_emb = self.model(pos)
        neg_emb = self.model(neg)

        loss_triplet = F.triplet_margin_loss(src_emb, pos_emb, neg_emb)

        self.log("train_loss", loss_triplet)

        return loss_triplet

    def validation_step(self, batch, batch_idx):
        src, pos, neg = batch["input"], batch["pos"], batch["neg"]
        src_emb = self.model(src)
        pos_emb = self.model(pos)
        neg_emb = self.model(neg)

        loss_triplet = F.triplet_margin_loss(src_emb, pos_emb, neg_emb)

        self.log("val_loss", loss_triplet)

        return loss_triplet

    def configure_optimizers(self):
        # Create Optimizer
        total_steps = self.config["num_train_steps"]
        gain_or_bias_params, rest_params = split_parameters_for_adamw(
            list(self.model.named_parameters())
        )
        # We will not apply weight decay to bias and layer norm parameters.
        optimizer = get_optimizer(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": self.config["optimizer"]["weight_decay"]},
            ],
            lr=self.config["optimizer"]["lr"],
            betas=(self.config["optimizer"]["beta1"], self.config["optimizer"]["beta2"]),
            eps=self.config["optimizer"]["eps"],
            momentum=self.config["optimizer"]["momentum"],
            optimizer_name=self.config["optimizer"]["optimizer"],
        )
        self.scheduler = cosine_lr(
            optimizer, self.config["optimizer"]["lr"], self.config["optimizer"]["warmup"], total_steps
        )

        return optimizer