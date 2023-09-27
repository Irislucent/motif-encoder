"""
From labeled midi files to readable dataset
This is for REAL labels, not generated labels
"""
import os
import glob
import pickle
import argparse
import shutil
from tqdm import tqdm

import numpy as np

from utils.midi_utils import *


def load_midi_labels(data_dir, labels_dir):
    # load pop909 data
    pop909_motive_label_dict = {}
    label_path_list = sorted(
        glob.glob(os.path.join(labels_dir, "*.mid"), recursive=True)
    )
    for label_path in tqdm(label_path_list):
        midi_id = int(os.path.splitext(os.path.basename(label_path))[0])
        tracks = read_midi_multitrack_unify_tempo(label_path)
        labels = tracks[0]
        # find and subtract beat onset
        midi_dir = os.path.join(data_dir, str(midi_id).zfill(3))
        midi_path = os.path.join(midi_dir, str(midi_id).zfill(3) + ".mid")
        beat_midi_path = os.path.join(midi_dir, "beat_midi.txt")
        with open(beat_midi_path, "r") as f:
            beat_midi = f.readlines()
        beat_onset = float(
            beat_midi[0].split()[0]
        )  # this is the global onset of this midi, in seconds
        labels[:, :2] -= beat_onset * get_midi_tempo(midi_path) / 60

        pop909_motive_label_dict[midi_id] = labels

    return pop909_motive_label_dict


def label_chunks(label_dict, chunk_len, chunks_dir, output_dir):
    """
    Args:
        label_dict: key: song_id, value: list of notes
        chunk_len: how many bars in every chunk
        chunks_dir: directory of chunks (preprocessed midi files)
        output_dir: directory of output
    """
    # create ouput_dir if necessary
    os.makedirs(output_dir, exist_ok=True)

    chunks_path_list = glob.glob(os.path.join(chunks_dir, "*.mid"))

    for song_id in list(label_dict.keys()):
        song_labels = label_dict[song_id]  # (number of notes, 4)
        # find corresponding chunks
        song_chunks_path_list = []
        for chunk_path in chunks_path_list:
            if int(os.path.basename(chunk_path).split("_")[0]) == song_id:
                song_chunks_path_list.append(chunk_path)
        song_chunks_path_list = sorted(song_chunks_path_list)
        song_chunk_ids = [
            os.path.splitext(os.path.basename(chunk_path))[0].split("_")[1]
            for chunk_path in song_chunks_path_list
        ]
        # register chunk labels
        motive_dict = {}  # key: motive_id (order of chunk_ids), value: list of paths
        existing_chunk_labels = []
        for chunk_id in song_chunk_ids:
            # find this chunk's path
            chunk_path = song_chunks_path_list[song_chunk_ids.index(chunk_id)]
            # read chunk label from song_labels
            chunk_label_raw = []  # in the form of: a list of notes representing motives
            slice_l = int(chunk_id) * chunk_len * 4
            slice_r = (int(chunk_id) + 1) * chunk_len * 4
            for note in song_labels:
                if (
                    note[0] < slice_l and note[1] > slice_l
                ):  # label starts before this chunk
                    if note[1] > slice_r:  # cut super long notes
                        chunk_label_raw.append(
                            np.array([0, chunk_len * 4, note[2], note[3]])
                        )
                    else:
                        chunk_label_raw.append(
                            np.array([0, note[1] - slice_l, note[2], note[3]])
                        )
                if (
                    note[0] >= slice_l and note[0] < slice_r
                ):  # labels starts in this chunk
                    if note[1] > slice_r:  # cut super long notes
                        chunk_label_raw.append(
                            np.array(
                                [note[0] - slice_l, chunk_len * 4, note[2], note[3]]
                            )
                        )
                    else:
                        chunk_label_raw.append(
                            np.array(
                                [note[0] - slice_l, note[1] - slice_l, note[2], note[3]]
                            )
                        )
                if note[0] > slice_r:  # label starts after this chunk
                    break
            chunk_label_raw = np.array(chunk_label_raw)
            if len(chunk_label_raw) == 0:  # totally no label
                continue
            chunk_label = summarize_chunk_label(
                chunk_label_raw, chunk_len
            )  # get a dominant midi number
            if chunk_label is None:  # no valid label
                continue
            if chunk_label not in existing_chunk_labels:  # seen label
                existing_chunk_labels.append(chunk_label)
            # name chunk labels according to order of appearance
            motive_id = existing_chunk_labels.index(
                chunk_label
            )  # turn midi number into a positive integer
            # brand new motive found
            if motive_id not in motive_dict:
                motive_dict[motive_id] = []
                motive_dict[motive_id].append(chunk_path)
            # if motive_id already in motive_dict, need to deduplicate
            else:
                already_in = False
                for i in range(len(motive_dict[motive_id])):
                    if midi_pianoroll_identical(
                        motive_dict[motive_id][i], chunk_path, chunk_len=chunk_len
                    ):
                        already_in = True
                        break
                if not already_in:
                    motive_dict[motive_id].append(chunk_path)

        # for every song, use motive_dict to move and rename midi files
        for motive_id in list(motive_dict.keys()):
            chunk_paths = motive_dict[motive_id]
            for i, chunk_path in enumerate(chunk_paths):
                output_name = (
                    "_".join([str(song_id).zfill(3), str(motive_id).zfill(3), str(i).zfill(3)])
                    + ".mid"
                )
                output_path = os.path.join(output_dir, output_name)
                shutil.copyfile(chunk_path, output_path)
        print(  # print out the number of motives, and their number of appearances
            "Song {} done with {} motives: {}".format(
                song_id,
                len(motive_dict.keys()),
                [len(motive_dict[motive_id]) for motive_id in list(motive_dict.keys())],
            )
        )


def summarize_chunk_label(chunk_label_raw, chunk_len, thres=0.75):
    """
    Chunks without a label covering over 75% of its length will be discarded
    Args:
        chunk_label_raw: (number of notes, 4), [[onset, offset, pitch, velocity]]
        chunk_len: length of a chunk in bars
        thres: threshold of covered time proportion for recognizing a label
    """
    # First make sure there's no overlap
    chunk_label = chunk_label_raw[np.argsort(chunk_label_raw[:, 0])]
    for i in range(len(chunk_label) - 1):
        if chunk_label[i][1] > chunk_label[i + 1][0]:
            raise ValueError("Overlap detected in chunk label")
    # Then count the total length of every midi number
    midi_num_dict = {}
    total_length = 4 * chunk_len
    for note in chunk_label:
        midi_num = note[2]
        if midi_num not in midi_num_dict:
            midi_num_dict[midi_num] = 0
        midi_num_dict[midi_num] += note[1] - note[0]
    # Finally, check if any midi number has a length over 75%
    for midi_num in list(midi_num_dict.keys()):
        if midi_num_dict[midi_num] / total_length > thres:
            return midi_num
    return None


def midi_pianoroll_identical(midi_path_1, midi_path_2, chunk_len, grid=1 / 32):
    """
    Check if two midi files have identical pianorolls
    Only single-track midi files are supported
    """
    pr_1 = read_midi_to_pianoroll(
        midi_path_1, grid=grid, pianoroll_len=int(chunk_len / grid)
    )
    pr_2 = read_midi_to_pianoroll(
        midi_path_2, grid=grid, pianoroll_len=int(chunk_len / grid)
    )
    if pr_1.shape != pr_2.shape:
        return False
    return np.all(pr_1 == pr_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/POP909")
    parser.add_argument(
        "--labels_dir", type=str, default="../data/POP909_motive_labels"
    )
    parser.add_argument(
        "--chunks_dir", type=str, default="../data/POP909_preprocessed_2bar"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/POP909_realmotives_2bar"
    )
    parser.add_argument("--chunk_len", type=int, default=1)
    args = parser.parse_args()

    # load pop909 data
    pop909_motive_label_dict = load_midi_labels(args.data_dir, args.labels_dir)

    label_chunks(
        label_dict=pop909_motive_label_dict,
        chunk_len=args.chunk_len,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
    )