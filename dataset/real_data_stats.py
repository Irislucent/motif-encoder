import os
import glob

import numpy as np


def get_stats(data_dir):
    print("Analysing data in {}".format(data_dir))
    midi_paths = glob.glob(os.path.join(data_dir, "*/*.mid"))
    song_motive_dict = {} # a dict of dicts
    for midi_path in midi_paths:
        basename = os.path.basename(midi_path)
        song_id = basename.split("_")[0]
        motive_id = basename.split("_")[1]
        if song_id in song_motive_dict.keys():
            if motive_id in song_motive_dict[song_id].keys():
                song_motive_dict[song_id][motive_id] += 1
            else:
                song_motive_dict[song_id][motive_id] = 1
        else:
            song_motive_dict[song_id] = {motive_id: 1}
    
    # compute mean motive number per song
    motive_num_list = []
    for song_id in song_motive_dict.keys():
        motive_num_list.append(len(song_motive_dict[song_id].keys()))
    print("Mean motive number per song: {}".format(np.mean(motive_num_list)))

    # compute mean number of appearances per motive
    motive_appearance_list = []
    for song_id in song_motive_dict.keys():
        for motive_id in song_motive_dict[song_id].keys():
            motive_appearance_list.append(song_motive_dict[song_id][motive_id])
    print("Mean number of appearances per motive: {}".format(np.mean(motive_appearance_list)))


if __name__ == '__main__':
    get_stats("../data/POP909_R_1bar")
