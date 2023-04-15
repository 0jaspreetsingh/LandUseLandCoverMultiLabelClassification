import argparse
import os

import numpy as np
import pandas as pd
from datadings.writer import FileWriter
from tqdm import tqdm
from tifffile import imread


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


def get_parser():
    parser = argparse.ArgumentParser(description='lclu-multispectral')
    parser.add_argument(
        '--src-dir',
        type=dir_path,
        help='Path to images and csv files')
    parser.add_argument(
        '--dest-dir',
        type=str,
        help='Path to write datadings file')
    return parser


def get_dataset(root_folder, file):
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    # csv_path = "/home/jsingh/Projects/LandCoverClassifier/multispectral-lulc/data/labels/csv"
    csv_path = "/home/jaspreet/Documents/DFKI/netscratch"
    csv_file = os.path.join(csv_path, file + ".csv")
    df = pd.read_csv(csv_file, sep=",", header=None)
    print(f"{file}.csv read successfully")
    images = np.asarray(df.iloc[:len(df), 0])
    lbs = np.asarray(df.iloc[:len(df), 1:])
    lbs = np.vstack(lbs).astype(np.int32)
    for i in tqdm(range(len(df))):
        sample = {'key': images[i], 'label': lbs[i]}
        for j in range(len(bands)):
            pth = os.path.join(root_folder, images[i], images[i] + "_" + bands[j] + ".tif")
            sample[bands[j]] = np.array(imread(pth), dtype=np.float32)
        yield sample


def main():
    parser = get_parser()
    args = parser.parse_args()
    root_folder = args.src_dir
    dest_folder = args.dest_dir
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    files = ["train", "test", "val"]
    for file in files:
        # path = os.path.join(dest_folder, file)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        print(f"Processing {file}.csv")
        msgpack_file = os.path.join(dest_folder, file + "_dataset.msgpack")
        with FileWriter(msgpack_file) as writer:
            for sample in get_dataset(root_folder, file):
                writer.write(sample)
        print(f"Processed {file}.csv successfully")


if __name__ == '__main__':
    print(__name__)
    main()
