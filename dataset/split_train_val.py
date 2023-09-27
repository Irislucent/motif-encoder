import os
import shutil
import argparse
import glob
from tqdm import tqdm


VAL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str,
    )
    parser.add_argument(
        "--save_dir", type=str
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.save_dir
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for midi_path in tqdm(glob.glob(os.path.join(data_dir, "*.mid"))):
        midi_name = os.path.basename(midi_path)
        midi_id = int(midi_name.split("_")[0])
        if midi_id in VAL_IDS:
            shutil.copy(midi_path, val_dir)
        else:
            shutil.copy(midi_path, train_dir)
