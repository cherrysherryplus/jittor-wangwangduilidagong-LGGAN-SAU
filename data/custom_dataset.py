"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

import os.path as osp


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        per_mode = 'resize_and_crop' if is_train else "fixed"
        parser.set_defaults(preprocess_mode=per_mode)
        load_size = 256 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(load_h=192)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(crop_h=192)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=29)
        parser.set_defaults(contain_dontcare_label=False)

        if is_train:
            label_dir = osp.join("./datasets/landscape", "labels")
        else:
            # test的input_path直接就是./testB/labels
            label_dir = "./datasets/landscape/testB/labels"
        parser.add_argument('--label_dir', type=str, default=label_dir,
                            help='path to the directory that contains label images')
        if is_train:
            image_dir = osp.join("./datasets/landscape", "imgs")
            parser.add_argument('--image_dir', type=str, default=image_dir ,
                                help='path to the directory that contains photo images')
        else:
            parser.add_argument('--image_dir', type=str, default=label_dir,
                                help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        input_path = opt.input_path
        if opt.phase == "train":
            label_dir = osp.join(input_path, "labels")
            image_dir = osp.join(input_path, "imgs")
        elif opt.phase == "test":
            label_dir = image_dir = input_path
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
