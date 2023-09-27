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

        # pass the batch to the model
        m1, m2 = batch["metaphor1"], batch["metaphor2"]
        m1_emb, m1_emb_expanded = self.model(m1)
        m2_emb, m2_emb_expanded = self.model(m2)
        
        # the joint embeddings should be close
        loss_inv = F.mse_loss(m1_emb_expanded, m2_emb_expanded)

        # the variance inside a batch is encouraged to be large (close to 1)
        m1_resid = m1_emb_expanded - m1_emb_expanded.mean(dim=0)
        m2_resid = m2_emb_expanded - m2_emb_expanded.mean(dim=0)

        m1_std = torch.sqrt(m1_resid.var(dim=0) + 0.0001)
        m2_std = torch.sqrt(m2_resid.var(dim=0) + 0.0001)

        loss_var = torch.mean(F.relu(1 - m1_std)) / 2 + torch.mean(F.relu(1 - m2_std)) / 2

        # the off-diagonal coefficients in the cov matrix are pushed to 0 to decorrelate the dimensions of the embeddings
        m1_cov = m1_resid.T @ m1_resid / (m1_resid.shape[0] - 1)
        m2_cov = m2_resid.T @ m2_resid / (m2_resid.shape[0] - 1)
        loss_cov = PLWrapper.off_diagonal(m1_cov).pow_(2).sum().div(m1_cov.shape[1]) \
            + PLWrapper.off_diagonal(m2_cov).pow_(2).sum().div(m2_cov.shape[1])

        # total loss
        train_loss = loss_inv * self.config["weight_inv"]  \
            + loss_var * self.config["weight_var"] \
            + loss_cov * self.config["weight_cov"]

        self.log("train_loss_inv", loss_inv)
        self.log("train_loss_var", loss_var)
        self.log("train_loss_cov", loss_cov)
        self.log("train_loss", train_loss)

        return train_loss
    
    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        # used to compute the covariance loss
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def validation_step(self, batch, batch_idx):
        # pass the batch to the model
        m1, m2 = batch["metaphor1"], batch["metaphor2"]
        m1_emb, m1_emb_expanded = self.model(m1)
        m2_emb, m2_emb_expanded = self.model(m2)
        
        # the joint embeddings should be close
        loss_inv = F.mse_loss(m1_emb_expanded, m2_emb_expanded)

        # the variance inside a batch is encouraged to be large (close to 1)
        m1_resid = m1_emb_expanded - m1_emb_expanded.mean(dim=0)
        m2_resid = m2_emb_expanded - m2_emb_expanded.mean(dim=0) # (batch_size, dim)

        m1_std = torch.sqrt(m1_resid.var(dim=0) + 0.0001)
        m2_std = torch.sqrt(m2_resid.var(dim=0) + 0.0001) # (dim,)

        loss_var = torch.mean(F.relu(1 - m1_std)) / 2 + torch.mean(F.relu(1 - m2_std)) / 2

        # the off-diagonal coefficients in the cov matrix are pushed to 0 to decorrelate the dimensions of the embeddings
        m1_cov = m1_resid.T @ m1_resid / (m1_resid.shape[0] - 1) # (dim, dim)
        m2_cov = m2_resid.T @ m2_resid / (m2_resid.shape[0] - 1)
        loss_cov = PLWrapper.off_diagonal(m1_cov).pow_(2).sum().div(m1_cov.shape[1]) \
            + PLWrapper.off_diagonal(m2_cov).pow_(2).sum().div(m2_cov.shape[1])

        # total loss
        val_loss = loss_inv * self.config["weight_inv"]  \
            + loss_var * self.config["weight_var"] \
            + loss_cov * self.config["weight_cov"]

        self.log("val_loss_inv", loss_inv)
        self.log("val_loss_var", loss_var)
        self.log("val_loss_cov", loss_cov)
        self.log("val_loss", val_loss)

        return val_loss

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