"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from PIL import Image
import os
import os.path as osp
from jittor.dataset import Dataset
import jittor.transform as TF
from data.image_folder import make_dataset


class FidDataset(Dataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """
    def __init__(self, opt) -> None:
        super(FidDataset, self).__init__()
        self.opt = opt
        self.labels, self.imgs = self.get_paths()

    def get_paths(self):
        train_input_path = self.opt.input_path
        # 当input_path为"./a/b/c//"即以'/'为结尾的路径类型时，先将末尾的'/'全都去掉
        if train_input_path[-1] == '/':
            train_input_path = osp.split(train_input_path)[0]
        # train_input_path此时为"./a/b/c"
        val_input_path = osp.join( osp.dirname(train_input_path), "val" ) 
        # 设置label_dir和image_dir
        label_dir = osp.join(val_input_path, "labels")
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        image_dir = osp.join(val_input_path, "imgs")
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        return label_paths, image_paths

    def __getitem__(self, index):
        label_paths, img_paths = self.labels, self.imgs
        labels = Image.open(label_paths[index])
        imgs = Image.open(img_paths[index])
        labels = TF.resize(labels, size=(192,256), interpolation=Image.NEAREST)
        labels = TF.to_tensor(labels)
        imgs = TF.resize(imgs, size=(192,256), interpolation=Image.BICUBIC)
        imgs = TF.to_tensor(imgs)
        return {'label': labels,'image': imgs}

    def __len__(self):
        return len(self.labels)

        