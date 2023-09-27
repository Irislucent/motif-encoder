"""
Run inference and visualization for an input song
Calls analyse/analyser.py, load a model and performs inference
"""
import os
import argparse
import yaml
import numpy as np
import torch
from analyse.analyser import MotiveAnalyser

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="bert.yaml",
        type=str
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str
    )
    args = parser.parse_args()
    
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    if args.checkpoint_path is not None:
        config["active_checkpoint"] = args.checkpoint_path # for inference

    ma = MotiveAnalyser(config, args.input_path)
    ma.load_model()
    ma.encode()
    ma.perform_clustering()
    ma.plot_colored_pr()
    ma.plot_hm()
    
    
    