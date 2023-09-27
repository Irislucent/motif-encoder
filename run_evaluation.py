"""
Evaluation for the models
Currently only on the synthesized dataset
Load the dataset and compute the recall @ k metric
"""
import os
import abc
import time
import argparse
import yaml
from glob import glob
from tqdm import tqdm

import numpy as np
import torch

from utils.midi_utils import *


class MotiveEvaluator():
    def __init__(self, config) -> None:
        self.data_dir = config["data_dir"]
        self.chunk_len = config["chunk_len"]
        self.grid = 1 / 32
        self.debug = config["debug"]
        self.config = config

        midi_paths = glob(os.path.join(self.data_dir, 'val', '*.mid'))

        self.midi_ids = []
        self.loaded_data = {}
        self.song_ids = []
        self.song_chunk_ids = {}
        print("Loading midis......")
        for midi_path in tqdm(midi_paths):
            midi_id = os.path.splitext(os.path.basename(midi_path))[0]
            song_id, chunk_id, mtp_id = midi_id.split("_")
            if self.debug:
                if int(chunk_id) != 34:
                    continue
            pr = read_midi_to_pianoroll(midi_path, grid=self.grid, pianoroll_len=int(self.chunk_len/self.grid))
            pr_repr = torch.tensor(pr.T, dtype=torch.float32) # (num of time grids, 84)
            # add data to loaded data and a lot of things...
            self.loaded_data[midi_id] = pr_repr
            self.midi_ids.append(midi_id)
            if song_id not in self.song_ids:
                self.song_ids.append(song_id)
            if song_id not in self.song_chunk_ids.keys():
                self.song_chunk_ids[song_id] = [chunk_id]
            else:
                self.song_chunk_ids[song_id].append(chunk_id)

        # count how many individual samples in total
        self.n_samples = len(self.loaded_data)

        # count how many songs in total
        self.n_songs = len(self.song_ids)

    def load_model(self):
        """
        Load model from checkpoint
        """
        config = self.config

        if config["method"] == "pianoroll_baseline":
            from pianoroll_baseline.bl import normalize_pr, original_pr
            if config["encoder"] == "normalized_pianoroll":
                self.encode_fn = normalize_pr
            elif config["encoder"] == "original_pianoroll":
                self.encode_fn = original_pr
            return
        elif config["method"] == "contrastive":
            from contrastive.train import PLWrapper
            if config["encoder"] == "bert":
                from contrastive.bert import BertEncoder
                model = BertEncoder(config["bert_config"])
        elif config["method"] == "regularized":
            from regularized.train import PLWrapper
            if config["encoder"] == "bert":
                from regularized.bert import BertEncoder
                model = BertEncoder(config["bert_config"])
            
        self.model = PLWrapper(model, config)
        checkpoint = torch.load(config["active_checkpoint"])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.encode_fn = self.model.forward


    def encode_all(self):
        """
        Encode all the data points
        """
        print("Encoding all the data points......")
        self.all_encodings = {}
        if self.config["method"] == "pianoroll_baseline":
            for midi_id in tqdm(self.midi_ids):
                self.all_encodings[midi_id] = self.encode_fn(self.loaded_data[midi_id].unsqueeze(0))[0].detach()
        elif self.config["method"] == "contrastive":
            for midi_id in tqdm(self.midi_ids):
                self.all_encodings[midi_id] = self.encode_fn(self.loaded_data[midi_id].unsqueeze(0))[0].detach()
        elif self.config["method"] == "regularized":
            for midi_id in tqdm(self.midi_ids):
                self.all_encodings[midi_id] = self.encode_fn(self.loaded_data[midi_id].unsqueeze(0))[0][0].detach()

    def precision_recall_at_k(self, input_midi_id, k_list=[5, 10, 20, 50, 100]):
        """
        Find the nearest k samples to the input sample
        Compute the precision & recall at k metrics using a list of positive samples
        Also compute the F1 score
        Return the precision, recall, F1 score at k
        """
        input_encoding = self.encode_fn(self.loaded_data[input_midi_id].unsqueeze(0))[0]

        # find all the positive samples
        pos_midi_ids = [] # positive samples are metaphors of the same chunk of the same song
        for this_midi_id in self.midi_ids:
            if this_midi_id.split("_")[:2] == input_midi_id.split("_")[:2] and this_midi_id != input_midi_id:
                pos_midi_ids.append(this_midi_id)
        if len(pos_midi_ids) == 0: # solitary motive
            return None, None, None

        # compute distances from the input to all the samples
        all_distances = {}
        for midi_id in self.midi_ids:
            if midi_id == input_midi_id:
                continue
            d = torch.norm(input_encoding - self.all_encodings[midi_id])
            all_distances[midi_id] = d

        # sort the distances
        all_distances = sorted(all_distances.items(), key=lambda x: x[1])
        # print(len(all_distances))

        # compute the precision & recall at k metrics
        precision_list = []
        recall_list = []
        f1_list = []
        for k in k_list:
            # find the nearest k samples
            nearest_k = all_distances[:k]
            nearest_k = [x[0] for x in nearest_k]

            # compute the precision at k metric
            precision = 0
            for midi_id in nearest_k:
                if midi_id in pos_midi_ids:
                    precision += 1
            precision /= k
            precision_list.append(precision)

            # compute the recall at k metric
            recall = 0
            for midi_id in nearest_k:
                if midi_id in pos_midi_ids:
                    recall += 1
            recall /= len(pos_midi_ids)
            recall_list.append(recall)

            # compute the f1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_list.append(f1)

        return precision_list, recall_list, f1_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="bert.yaml",
        type=str
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str
    )
    args = parser.parse_args()
    
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    if args.checkpoint_path is not None:
        config["active_checkpoint"] = args.checkpoint_path # for inference

    me = MotiveEvaluator(config)
    me.load_model()
    me.encode_all()

    k_list = [1, 5, 10, 20, 50, 100, 200, 500]
    mean_precision_at_ks = np.zeros(len(k_list))
    mean_recall_at_ks = np.zeros(len(k_list))
    # mean_f1_at_ks = np.zeros(len(k_list))
    cnt_inputs = 0
    for midi_id in tqdm(me.midi_ids):
        precision_at_ks, recall_at_ks, f1_at_ks = me.precision_recall_at_k(midi_id, k_list)
        if precision_at_ks is None and recall_at_ks is None and f1_at_ks is None:
            continue
        cnt_inputs += 1
        mean_precision_at_ks = np.array(precision_at_ks) + mean_precision_at_ks
        mean_recall_at_ks = np.array(recall_at_ks) + mean_recall_at_ks
        # mean_f1_at_ks = np.array(f1_at_ks) + mean_f1_at_ks
    mean_precision_at_ks /= cnt_inputs
    mean_recall_at_ks /= cnt_inputs
    # mean_f1_at_ks /= cnt_inputs
    for k in range(len(k_list)):
        print(f"Mean Precision@K, K = {k_list[k]}", mean_precision_at_ks[k])
        print(f"Mean Recall@K, K = {k_list[k]}", mean_recall_at_ks[k])
        # print(f"Mean F1@K, K = {k_list[k]}", mean_f1_at_ks[k])