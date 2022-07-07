"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from jittor.dataset import Dataset
import jittor.transform as TF
from data.image_folder import make_dataset
import argparse 
from PIL import Image


class FidDataset(Dataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """
    def __init__(self) -> None:
        super(FidDataset, self).__init__()
        self.labels, self.imgs = self.get_paths()
        # self.opt = self.modify_commandline_options()

    def get_paths(self):
        label_dir = './datasets/landscape/val/labels'
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = './datasets/landscape/val/imgs'
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        return label_paths, image_paths

    def __getitem__(self, index):
        label_paths, img_paths = self.labels, self.imgs
        labels = Image.open(label_paths[index])
        imgs = Image.open(img_paths[index])
        labels = TF.resize(labels, size=(384,512), interpolation=Image.NEAREST)
        labels = TF.to_tensor(labels)
        imgs = TF.resize(imgs, size=(384,512), interpolation=Image.BICUBIC)
        imgs = TF.to_tensor(imgs)
        return {'label': labels,'image': imgs}

    def __len__(self):
        return len(self.labels)

        