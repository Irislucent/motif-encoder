import os
import glob
import argparse
from tqdm import tqdm

import numpy as np

from utils.midi_utils import *


def load_midi_data(data_dir):
    # load pop909 data
    pop909_data_dict = {}
    midi_path_list = sorted(
        glob.glob(os.path.join(data_dir, "*/*.mid"), recursive=True)
    )
    for midi_path in tqdm(midi_path_list):
        midi_dir = os.path.dirname(midi_path)
        midi_id = int(os.path.splitext(os.path.basename(midi_path))[0])
        tracks = read_midi_multitrack_unify_tempo(midi_path)
        acc = tracks[2] # in this study, we only use the accompaniment track
        # find and subtract beat onset
        beat_midi_path = os.path.join(midi_dir, "beat_midi.txt")
        with open(beat_midi_path, "r") as f:
            beat_midi = f.readlines()
        beat_onset = float(
            beat_midi[0].split()[0]
        )  # this is the global onset of this midi, in seconds
        acc[:, :2] -= beat_onset * get_midi_tempo(midi_path) / 60
        acc = quantize(acc, quantize_offset=True)

        pop909_data_dict[midi_id] = acc

    return pop909_data_dict


def chunk_data(data_dict, chunk_len=1):
    """
    chunk_len: how many bars in every chunk
    """
    chunked_data_dict = {}
    for key, notes in data_dict.items():
        chunk_id = 0
        chunks = [[]]
        for i, note in enumerate(notes):
            if (
                note[0] >= chunk_id * chunk_len * 4
                and note[0] < (chunk_id + 1) * chunk_len * 4
            ):
                chunks[-1].append(
                    np.array(
                        [
                            note[0] - chunk_id * chunk_len * 4,
                            note[1] - chunk_id * chunk_len * 4,
                            note[2],
                            note[3],
                        ]
                    )
                )
            else:
                chunks.append([])
                chunk_id += 1
                chunks[-1].append(
                    np.array(
                        [
                            note[0] - chunk_id * chunk_len * 4,
                            note[1] - chunk_id * chunk_len * 4,
                            note[2],
                            note[3],
                        ]
                    )
                )
        for i in range(len(chunks)):
            chunked_data_dict[str(key).zfill(3) + "_" + str(i).zfill(3)] = chunks[i]
    return chunked_data_dict


def quantize(notes, grid=1 / 32, quantize_offset=True):
    """
    Quantize a list of notes like [[start, end, pitch, velocity]]
    Here the grid means "resolution in one beat"
    However, this quantization will be overwritten by pretty_midi.instruments.get_piano_roll() anyways
    """
    notes_q = []
    grids = np.arange(1 / grid) * grid
    for i, note in enumerate(notes):
        # onset
        frac = note[0] - int(note[0])
        frac_q = grids[np.argmin(np.abs(grids - frac))]
        onset_q = int(note[0]) + frac_q
        # offset
        if quantize_offset:
            frac = note[1] - int(note[1])
            frac_q = grids[np.argmin(np.abs(grids - frac))]
            offset_q = int(note[1]) + frac_q
        else:
            offset_q = note[1]

        notes_q.append(np.array([onset_q, offset_q, note[2], note[3]]))

    return np.array(notes_q)


def save_processed_data(data_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for key, notes in data_dict.items():
        save_path = os.path.join(save_dir, str(key) + ".mid")
        write_midi(save_path, notes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/POP909",
        help="POP909 data dir",
    )
    parser.add_argument(
        "--save_dir", type=str, default="../data/POP909_preprocessed_1bar"
    )
    args = parser.parse_args()

    data_dict = load_midi_data(args.data_dir)
    chunked_data_dict = chunk_data(data_dict, chunk_len=1)
    save_processed_data(chunked_data_dict, args.save_dir)
