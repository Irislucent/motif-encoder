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
import pretty_midi
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.midi_utils import *
from pianoroll_baseline.bl import normalize_pr


class MetaphorTripletDataset(Dataset):
    """
    Dataset for triplet loss
    """
    def __init__(self, mtp_dir, chunk_len=1, grid=1 / 32, neg_enhance=False, debug=False):
        super(MetaphorTripletDataset, self).__init__()
        mtp_midi_paths = glob.glob(os.path.join(mtp_dir, "*.mid"))

        self.midi_ids = []
        self.loaded_data = {}
        self.song_ids = []
        self.song_chunks = {} # key: song_id, value: dict of {chunk_id: n_metaphors}
        self.chunk_len = chunk_len
        self.neg_enhance = neg_enhance # if true, will filter out the negative samples that are too similar to the anchor

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

        # count how many individual samples in total
        self.n_samples = len(self.loaded_data)

        # count how many songs in total
        self.n_songs = len(self.song_ids)

        self.n_neg_enhanced = 0

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

        # when using the synthesized dataset, negative samples are from different songs
        if self.neg_enhance == False:
            neg_song_id = self.song_ids[
                (int(input_song_id) + random.randint(0, self.n_songs - 1)) % self.n_songs
            ]
            neg_chunk_id = random.choice(list(self.song_chunks[neg_song_id].keys()))
            n_mtp = self.song_chunks[neg_song_id][neg_chunk_id]
            neg_mtp_id = random.randint(0, n_mtp - 1)
            neg_midi_id = "_".join([neg_song_id, neg_chunk_id, str(neg_mtp_id).zfill(3)])

            pos_sample = self.loaded_data[pos_midi_id]
            neg_sample = self.loaded_data[neg_midi_id]
        else:
            # filter out some "bad" negative samples that are too similar to the anchor
            # similar rate: the ratio of the sum of the element-wise product of the anchor and the negative sample to the sum of the anchor
            # let's say the threshold is 50%
            while True:
                neg_song_id = self.song_ids[
                    (int(input_song_id) + random.randint(0, self.n_songs - 1)) % self.n_songs
                ]
                neg_chunk_id = random.choice(list(self.song_chunks[neg_song_id].keys()))
                n_mtp = self.song_chunks[neg_song_id][neg_chunk_id]
                neg_mtp_id = random.randint(0, n_mtp - 1)
                neg_midi_id = "_".join([neg_song_id, neg_chunk_id, str(neg_mtp_id).zfill(3)])

                pos_sample = self.loaded_data[pos_midi_id]
                neg_sample = self.loaded_data[neg_midi_id]

                normalized_input_sample = normalize_pr(input_sample, batched=False)
                normalized_neg_sample = normalize_pr(neg_sample, batched=False)
                similar_rate = torch.sum(normalized_input_sample * normalized_neg_sample) / torch.sum(normalized_input_sample)

                if similar_rate < 0.5 or torch.sum(normalized_input_sample) == 0:
                    break
                self.n_neg_enhanced += 1

        data_item = {"input": input_sample, "pos": pos_sample, "neg": neg_sample}

        return data_item


def collate_fn(batch, pad_value=0):
    batch_size = len(batch)
    input_seq = [data_item["input"] for data_item in batch]
    pos_seq = [data_item["pos"] for data_item in batch]
    neg_seq = [data_item["neg"] for data_item in batch]

    triple_seq = input_seq
    triple_seq.extend(pos_seq)
    triple_seq.extend(neg_seq)
    triple_seq = pad_sequence(triple_seq, batch_first=True, padding_value=pad_value)
    input_seq = triple_seq[0:batch_size]
    pos_seq = triple_seq[batch_size : batch_size * 2]
    neg_seq = triple_seq[batch_size * 2 :]
    # input_seq = pad_sequence(input_seq, batch_first=True, padding_value=pad_value)
    # pos_seq = pad_sequence(pos_seq, batch_first=True, padding_value=pad_value)
    # neg_seq = pad_sequence(neg_seq, batch_first=True, padding_value=pad_value)
    out_batch = {"input": input_seq, "pos": pos_seq, "neg": neg_seq}

    return out_batch


def get_dataloader(mtp_dir, chunk_len=2, batch_size=1, num_workers=0, shuffle=True, neg_enhance=True, debug=False):
    dataset = MetaphorTripletDataset(mtp_dir, chunk_len=chunk_len, neg_enhance=neg_enhance, debug=debug)
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
    dl = get_dataloader(args.data_dir, args.chunk_len, batch_size=16, neg_enhance=True, debug=False)

    total_size = 0
    for batch in dl:
        print(batch["input"].shape)
        print(batch["pos"].shape)
        print(batch["neg"].shape)
        print(sys.getsizeof(batch))
        total_size += sys.getsizeof(batch)

    a = 1