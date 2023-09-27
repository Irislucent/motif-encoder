import os
import sys
import logging
import datetime
import glob
import pickle
import argparse
import yaml
import random
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from utils.midi_utils import *


def main(config):
    # set logs
    os.makedirs(config["log_dir"], exist_ok=True)
    if config["name"] is None:
        config["name"] = datetime.datetime.now().strftime("%m-%d-%H-%M")
    log_base_dir = os.path.join(config["log_dir"], config["name"])
    os.makedirs(log_base_dir, exist_ok=True)

    if config["debug"]:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt="%d-%M-%Y %H:%M:%S",
        level=log_level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_base_dir, "main.log")),
        ],
    )

    # set method used
    if config["method"] == "contrastive":
        from contrastive.bert import BertEncoder
        from contrastive.dataloader import get_dataloader
        from contrastive.train import PLWrapper
    elif config["method"] == "regularized":
        from regularized.bert import BertEncoder
        from regularized.dataloader import get_dataloader
        from regularized.train import PLWrapper

    # set dataloader
    if config["method"] == "contrastive":
        train_dl = get_dataloader(
            os.path.join(config["data_dir"], "train"),
            chunk_len=config["chunk_len"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            shuffle=True,
            neg_enhance=config["neg_enhance"],
            debug=config["debug"],
        )
        val_dl = get_dataloader(
            os.path.join(config["data_dir"], "val"),
            chunk_len=config["chunk_len"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            shuffle=False,
            neg_enhance=config["neg_enhance"],
            debug=config["debug"],
        )
    elif config["method"] == "regularized":
        train_dl = get_dataloader(
            os.path.join(config["data_dir"], "train"),
            chunk_len=config["chunk_len"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            shuffle=True,
            debug=config["debug"],
        )
        val_dl = get_dataloader(
            os.path.join(config["data_dir"], "val"),
            chunk_len=config["chunk_len"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            shuffle=False,
            debug=config["debug"],
        )
    config["num_train_steps"] = len(train_dl) * config["epochs"]
    config["num_train_batch_per_epoch"] = len(train_dl)

    # create model
    if config["encoder"] == "bert":
        model = BertEncoder(config["bert_config"])

    # Print model parameters and hyperparameters
    logging.info("Model:")
    logging.info(f"{str(model)}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")
    logging.info("HyperParams:")
    cfg_file = os.path.join(log_base_dir, "config.yaml")
    with open(cfg_file, "w") as f:
        f.write(yaml.dump(config))

    # Wandb logger
    wandb_logger = WandbLogger(
        project="metaphor",
        notes=config["name"],
        name=config["name"],
        tags=[],
        config=config,
    )
    wandb.save(cfg_file)

    # Checkpoint save callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=5,
        every_n_epochs=config["save_frequency"],
        dirpath=os.path.join(log_base_dir, "checkpoints"),
        filename="checkpoint-{epoch:02d}",
    )

    # Learning rate callback
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # set pl trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["epochs"],
        val_check_interval=config["val_check_interval"],
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
    )

    # freeze part of the model if fine-tuning
    if "pretrained_method" in config and config["pretrained_method"] == "regularized":
        from regularized.train import PLWrapper as pt_PLWrapper
        from regularized.bert import BertEncoder as pt_BertEncoder
        model_pt = pt_BertEncoder(config["bert_config"])
        # This is a hacky way to load a Lightning checkpoint
        # Have to load it this way because we passed nn.Module into LightningModule as a hyperparameter
        # But this is the best workaround I can find
        model_pt = pt_PLWrapper.load_from_checkpoint(config["load_checkpoint"], model=model_pt, config=config)
        state_dict = model_pt.model.state_dict()
        model = PLWrapper(model, config)
        model.model.load_state_dict(state_dict, strict=False)
        trainer.fit(model, train_dl, val_dl)
    else:
        # set pl training wrapper
        model = PLWrapper(model, config)
    
    trainer.fit(model, train_dl, val_dl, ckpt_path=config["load_checkpoint"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="bert_config.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)
