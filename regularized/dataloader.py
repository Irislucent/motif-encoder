import os
import sys
import time
import logging
import datetime
import glob
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.midi_utils import *


class MetaphorJointEmbeddingDataset(Dataset):
    """
    Dataset for VICReg
    """
    def __init__(self, mtp_dir, chunk_len=1, grid=1 / 32, debug=False):
        super(MetaphorJointEmbeddingDataset, self).__init__()
        mtp_midi_paths = glob.glob(os.path.join(mtp_dir, "*.mid"))

        self.midi_ids = []
        self.loaded_data = {}
        self.song_ids = []
        self.song_chunks = {} # key: song_id, value: dict of {chunk_id: n_metaphors}
        self.chunk_len = chunk_len
        print("Loading midis......")
        for midi_path in tqdm(mtp_midi_paths):
            midi_id = os.path.splitext(os.path.basename(midi_path))[0]
            song_id, chunk_id, mtp_id = midi_id.split("_")
            if debug:
                if int(chunk_id) != 34:
                    continue
            pr = read_midi_to_pianoroll(midi_path, grid=grid, pianoroll_len=int(self.chunk_len/grid))
            pr_repr = torch.tensor(pr.T, dtype=torch.float32) # (num of time grids, 84)
            # add data to loaded data and a lot of things...
            self.loaded_data[midi_id] = pr_repr
            self.midi_ids.append(midi_id)
            if song_id not in self.song_ids: # new song
                self.song_ids.append(song_id)
            if song_id not in self.song_chunks.keys(): # new song
                self.song_chunks[song_id] = {chunk_id: 1}
            else:
                if chunk_id not in [x[0] for x in self.song_chunks[song_id].items()]: # new chunk of existing song
                    self.song_chunks[song_id][chunk_id] = 1
                else: # chunk_id already in the list
                    self.song_chunks[song_id][chunk_id] += 1 # number of metaphors in this chunk will add one

        # need to remove motives with only one metaphor
        midi_ids_old = self.midi_ids.copy()
        for midi_id in midi_ids_old:
            song_id, chunk_id, mtp_id = midi_id.split("_")
            if self.song_chunks[song_id][chunk_id] == 1:
                self.midi_ids.remove(midi_id)
                del self.loaded_data[midi_id]
                del self.song_chunks[song_id][chunk_id]

        # count how many individual samples in total
        self.n_samples = len(self.loaded_data)

        # count how many songs in total
        self.n_songs = len(self.song_ids)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        input_midi_id = self.midi_ids[idx]
        input_sample = self.loaded_data[input_midi_id]
        input_song_id, input_chunk_id, input_mtp_id = input_midi_id.split("_")

        # positive samples are from the same chunk but different metaphor
        n_mtp = self.song_chunks[input_song_id][input_chunk_id]
        pos_mtp_id = (
            int(input_mtp_id) + random.randint(0, n_mtp - 1)
        ) % n_mtp # computed like this to avoid the anchor itself
        pos_midi_id = "_".join(
            [input_song_id, input_chunk_id, str(pos_mtp_id).zfill(3)]
        )

        pos_sample = self.loaded_data[pos_midi_id]

        data_item = {"metaphor1": input_sample, "metaphor2": pos_sample}

        return data_item


def collate_fn(batch, pad_value=0):
    batch_size = len(batch)
    m1_seq = [data_item["metaphor1"] for data_item in batch]
    m2_seq = [data_item["metaphor2"] for data_item in batch]

    double_seq = m1_seq
    double_seq.extend(m2_seq)
    double_seq = pad_sequence(double_seq, batch_first=True, padding_value=pad_value)
    m1_seq = double_seq[0:batch_size]
    m2_seq = double_seq[batch_size : batch_size * 2]
    out_batch = {"metaphor1": m1_seq, "metaphor2": m2_seq}

    return out_batch


def get_dataloader(mtp_dir, chunk_len=2, batch_size=1, num_workers=0, shuffle=True, debug=False):
    dataset = MetaphorJointEmbeddingDataset(mtp_dir, chunk_len=chunk_len, debug=debug)
    return DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/POP909_R_1bar/train",
        help="POP909 data dir",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=1 / 32,
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=1
    )

    args = parser.parse_args()
    dl = get_dataloader(args.data_dir, args.chunk_len, batch_size=16)

    total_size = 0
    for batch in dl:
        print(batch["metaphor1"].shape)
        print(batch["metaphor2"].shape)
        print(sys.getsizeof(batch))
        total_size += sys.getsizeof(batch)