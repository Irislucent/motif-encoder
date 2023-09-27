import os
import sys
import logging
import datetime
import glob
import random
import argparse
from tqdm import tqdm

import numpy as np

from utils.midi_utils import *
from utils.training_utils import set_logger


class MidiMetaphor:
    def __init__(self) -> None:
        """
        Midi Metaphor Converter (rules only for now)
        """
        pass

    def metaphorize_piece(
        self, src_midi_path, trg_dir, n_metaphors=10, accept_timesig=["4/4"], piece_len=2
    ):
        """
        piece_len: how many bars in every chunk
        """
        self.notes_raw = read_midi(src_midi_path)
        self.tempo = get_midi_tempo(src_midi_path)
        self.piece_id = os.path.splitext(os.path.basename(midi_path))[0]
        self.piece_len = piece_len

        # ignore empty chunks and any chunks with less than 3 notes
        if len(self.notes_raw) < 3:
            return

        for i in range(n_metaphors):
            if i == 0:  # The 0th metaphor is itself
                save_name = self.piece_id + "_" + str(i).zfill(3) + ".mid"
                write_midi(os.path.join(trg_dir, save_name), self.notes_raw)
            else:
                new_notes = self.notes_raw.copy()
                new_notes = self.last_note_dur_variation(new_notes)
                new_notes = self.random_dropout(new_notes)
                new_notes = self.random_move(new_notes)
                new_notes = self.random_transpose(new_notes)
                save_name = self.piece_id + "_" + str(i).zfill(3) + ".mid"
                write_midi(os.path.join(trg_dir, save_name), new_notes)

    def random_dropout(self, notes):
        drop_idx = random.randint(0, len(notes))
        if drop_idx == len(notes):  # no dropout
            return notes
        notes = np.delete(notes, drop_idx, axis=0)

        return notes

    def random_move(self, notes):
        shift_semitones = random.randint(-2, 2)
        mv_idx = random.randint(0, len(notes) - 1)
        notes[mv_idx][2] += shift_semitones
        return notes

    def random_transpose(self, notes):
        shift_semitones = random.randint(-6, 6)
        for i, note in enumerate(notes):
            notes[i][2] += shift_semitones
        return notes

    def last_note_dur_variation(self, notes):
        """
        From ThemeTransformer
        """
        last_note_idx = np.argmax(notes[:, 1])
        last_note_start = notes[last_note_idx][0]
        chunk_end = self.piece_len * 4
        if (chunk_end - last_note_start) * 4 < 1: # no space for variation
            return notes
        last_note_new_end = random.randint(1, int((chunk_end - last_note_start) * 4)) / 4 + last_note_start
        notes[last_note_idx][1] = last_note_new_end
        return notes

    def note_split_and_combine(self, notes):
        """
        From ThemeTransformer
        This is too slow. Not used.
        """
        new_notes = []
        note_idx = 0
        while note_idx < len(notes) - 1:
            # Find consecutive notes with same pitch
            if notes[note_idx][2] == notes[note_idx + 1][2]:
                # Combine these two with probability 0.5
                if random.randint(0, 1) == 0:
                    new_notes.append(
                        [
                            notes[note_idx][0],
                            notes[note_idx + 1][1],
                            notes[note_idx][2],
                            notes[note_idx][3],
                        ]
                    )
                    note_idx += 2
        notes = np.array(new_notes)
        new_notes = []
        while note_idx < len(notes):
            # randomly split some notes into two with p=0.05
            if random.randint(0, 20) == 0:
                dur = notes[note_idx][1] - notes[note_idx][0]
                new_notes.append(
                    [
                        notes[note_idx][0],
                        notes[note_idx][0] + dur / 2,
                        notes[note_idx][2],
                        notes[note_idx][3],
                    ]
                )
                new_notes.append(
                    [
                        notes[note_idx][0] + dur / 2,
                        notes[note_idx][1],
                        notes[note_idx][2],
                        notes[note_idx][3],
                    ]
                )
            else:
                new_notes.append(notes[note_idx])
                note_idx += 1

        return np.array(new_notes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/POP909_preprocessed_1bar",
        help="POP909 data dir (should have been preprocessed)",
    )
    parser.add_argument(
        "--piece_len",
        type=int,
        default=1,
        help="Piece length in bars"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/POP909_metaphors_1bar",
        help="POP909 data dir (should have been preprocessed)",
    )
    parser.add_argument(
        "--n_metaphors",
        type=int,
        default=6
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    m = MidiMetaphor()
    midi_paths = glob.glob(os.path.join(args.data_dir, "*.mid"))
    for midi_path in tqdm(midi_paths):
        m.metaphorize_piece(midi_path, args.save_dir, n_metaphors=args.n_metaphors, piece_len=args.piece_len)
    print(f"Total number of metaphors: {len(os.listdir(args.save_dir))}")
