"""
Break the target song into chunks
"""
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.midi_utils import *
from dataset.preprocess import quantize, chunk_data

class MotiveAnalyseDataset(Dataset):
    def __init__(self, midi_path, chunk_size=1, grid=1 / 32, representation='piano_roll'):
        """
        This dataset is only used for testing, so do not not not not not shuffle.
        Args:
            midi_path (str)     :   Path to the input .mid file.
            chunk_size (int)    :   The size of each chunk.
            grid (float)        :   The grid size of the piano roll. (nth note)
            representation (str):   The representation of the data. 'piano_roll' or 'geometric'
        """
        if representation not in ['piano_roll', 'geometric']:
            raise ValueError("representation must be 'piano_roll' or 'geometric'")
        elif representation == 'geometric':
            raise NotImplementedError("geometric representation is not implemented yet")
        elif representation == 'piano_roll':
            self.representation = 'piano_roll'
            self.grid = grid
            self.chunk_size = chunk_size
            self.midi_path = midi_path

        self.load_midi_file(self.midi_path)

    def load_midi_file(self, midi_path):
        """
        This loading logic only works for POP909.
        """
        # load midi to unified-tempo pianoroll
        tracks = read_midi_multitrack_unify_tempo(midi_path)
        acc = tracks[2] # in this study, we only use the accompaniment track
        # find and subtract beat onset
        midi_dir = os.path.dirname(midi_path)
        beat_midi_path = os.path.join(midi_dir, "beat_midi.txt")
        with open(beat_midi_path, "r") as f:
            beat_midi = f.readlines()
        beat_onset = float(
            beat_midi[0].split()[0]
        )  # this is the global onset of this midi, in seconds
        acc[:, :2] -= beat_onset * get_midi_tempo(midi_path) / 60
        # quantize
        acc = quantize(acc, quantize_offset=True)
        # chunk
        data_dict = {}
        data_dict[midi_path] = acc # don't really need this dict, but for consistency with the dataset
        chunked_data_dict = chunk_data(data_dict, chunk_len=self.chunk_size)
        data_chunks = [notes for key, notes in chunked_data_dict.items()]
        # turn every chunk to a pianoroll matrix
        self.data_chunks = []
        for i, chunk_notes in enumerate(data_chunks):
            if len(chunk_notes) == 0:
                self.data_chunks.append(torch.zeros((32, 84), dtype=torch.float32))
                continue
            # save the chunk to a tmp midi file
            tmp_midi_path = "tmp.mid"
            write_midi(tmp_midi_path, chunk_notes, bpm=60)
            # load the tmp midi file to pianoroll
            chunk_pr = read_midi_to_pianoroll(tmp_midi_path, grid=self.grid, pianoroll_len=int(self.chunk_size / self.grid))
            self.data_chunks.append(torch.tensor(chunk_pr.T, dtype=torch.float32))
            # remove the tmp midi file
            os.remove(tmp_midi_path)
        
    def __len__(self):
        return len(self.data_chunks)

    def __getitem__(self, idx):
        return self.data_chunks[idx]


def get_dataloader(midi_path, chunk_size=1, grid=1 / 32, representation='piano_roll'):
    dataset = MotiveAnalyseDataset(midi_path, chunk_size=chunk_size, grid=grid, representation=representation)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def test(midi_path):
    dataloader = get_dataloader(midi_path)
    for data_chunk in dataloader:
        assert data_chunk.shape[1] == 128
        assert data_chunk.shape[2] == 32

if __name__ == "__main__":
    midi_path = "../data/POP909/001/001.mid"
    test(midi_path)