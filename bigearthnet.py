import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from tifffile import imread


class BigEarthNetDataset(Dataset):
    def __init__(self, root_folder, csv_file, rgb=False):
        self.root_folder = root_folder
        df = pd.read_csv(os.path.join(root_folder, csv_file), sep=",", header=None)

        self.images = np.asarray(df.iloc[:len(df), 0])
        lbs = np.asarray(df.iloc[:len(df), 1:])
        lbs = np.vstack(lbs).astype(np.int32)
        self.labels = torch.from_numpy(lbs)
        self.rgb = rgb
        self.transforms = Compose([ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if not self.rgb:
            img_10m_path = os.path.join(self.root_folder, "tiffimages", str(self.images[index]),
                                        str(self.images[index]) + "_10m.tiff")
            img_10m = imread(img_10m_path)
            img_10m = self.transforms(img_10m)

            label = self.labels[index]
            return img_10m.float(), label.float()
        else:
            image = os.path.join(self.root_folder, "images", str(self.images[index]) + ".jpg")
            img = Image.open(image)
            img = self.transforms(img)
            label = self.labels[index]
            return img, label.float()