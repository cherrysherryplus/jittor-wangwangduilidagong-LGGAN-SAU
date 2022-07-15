"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from traitlets import default
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        
        # 输入输出路径
        parser.add_argument('--input_path', type=str, default='./datasets/landscape/testB/labels', help="输入label路径")
        parser.add_argument('--output_path', type=str, default='./results', help="输出image路径")
        # 加载latest或best模型，请使用latest
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        
        # 使用超分辨率缩放到384 * 512，默认为否
        parser.add_argument('--use_sr', action='store_true', help='默认使用Image.resize(Bicubic mode)缩放，\
                                                                  指定该参数后使用在比赛数据训练集上训练的RCAN模型缩放')

        # 输入label固定为192 * 256
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(crop_size=256, crop_h=192)
        parser.set_defaults(load_size=256, load_h=192, display_winsize=256)

        # dataset config
        parser.set_defaults(dataset_mode='custom')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(nThreads=1)
        parser.set_defaults(batchSize=1)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        
        # For generator: use sau module
        parser.set_defaults(use_sau=True)

        self.isTrain = False
        return parser

