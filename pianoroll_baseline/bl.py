"""
Baseline model
Using the pianoroll representation, but transpose all the pianorolls to the same "key"
So that we have a interval-based representation that starts from the lowest note 
interval representation
"""
import torch

def normalize_pr(pianoroll, batched=True):
    """
    Normalize the pianoroll to start from the lowest note
    E.g.
        [[[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], ...]] -> [[[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], ...]]
    Args:
        pianoroll: torch.tensor, (batch, num of time grids, 84)
        batched: bool, whether the pianoroll is batched at the first dimension or not
    """
    if batched:
        # find the lowest note
        lowest_pitch = 0
        for i in range(pianoroll.shape[2]):
            if pianoroll[:, :, i].sum() > 0:
                lowest_pitch = i
                break
        # transpose the pianoroll
        pianoroll = pianoroll.clone()
        pianoroll = torch.roll(pianoroll, -lowest_pitch, dims=2)
    else:
        # find the lowest note
        lowest_pitch = 0
        for i in range(pianoroll.shape[1]):
            if pianoroll[:, i].sum() > 0:
                lowest_pitch = i
                break
        # transpose the pianoroll
        pianoroll = pianoroll.clone()
        pianoroll = torch.roll(pianoroll, -lowest_pitch, dims=1)

    return pianoroll

def original_pr(pianoroll):
    return pianoroll