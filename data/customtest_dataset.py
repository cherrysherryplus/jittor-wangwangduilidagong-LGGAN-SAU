"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from PIL import Image

from data.base_dataset import get_params, get_transform
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomtestDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)

        # preprocess config
        parser.set_defaults(preprocess_mode="fixed")
        parser.set_defaults(load_size=512)
        parser.set_defaults(load_h=384)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(crop_h=384)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=29)
        parser.set_defaults(contain_dontcare_label=False)
        
        # data path config
        label_dir = './datasets/landscape/testB/labels'
        parser.add_argument('--label_dir', type=str, default=label_dir,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default=label_dir,
                                help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        instance_paths = []

        assert len(label_paths) == len(image_paths), f"The #images in {label_paths} and {image_paths} do not match. Is there something wrong?"
        return label_paths, image_paths, instance_paths

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)

        # input image (real images)
        image_path = self.image_paths[index]
        image_tensor = jt.zeros_like(label_tensor).repeat([1,3,1,1])

        input_dict = {'label0': label_tensor[:,0:192,0:256],
                      'label1': label_tensor[:,0:192,256:512],
                      'label2': label_tensor[:,192:384,:256],
                      'label3': label_tensor[:,192:384,256:512],
                      'image': image_tensor,
                      'path': image_path,
                      }

        self.postprocess(input_dict)
        return input_dict


if __name__ == "__main__":
    test_dataset = CustomtestDataset()
    from options.test_options import TestOptions
    opt = TestOptions().parse()
    dataloader = test_dataset.set_attrs(batch_size=1,
            shuffle=True,
            num_workers=0,
            drop_last=False)
    dataloader.initialize(opt)
    print("the testDataset is contain %d labels" %(len(dataloader)))

    for i, data_i in enumerate(dataloader):
        img_path = data_i['path']
        print(data_i['label0'].shape)
        print(data_i['label1'].shape)
        print(data_i['label2'].shape)
        print(data_i['label3'].shape)
        