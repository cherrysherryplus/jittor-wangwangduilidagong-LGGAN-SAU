"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from pyexpat import model
import jittor as jt
import PIL.Image as Image

import os
import os.path as osp
from pathlib import Path
from shutil import rmtree
from tqdm import tqdm
from collections import OrderedDict

from options.test_options import TestOptions
from util.util import fix_seed


# Label to Image
def label_to_image(opt):
    # dataset
    import data
    dataset = data.create_dataloader(opt)
    dataloader_generate = dataset().set_attrs(batch_size=4,
                                    shuffle=not opt.serial_batches,
                                    num_workers=int(opt.nThreads),
                                    drop_last=opt.isTrain)
    dataloader_generate.initialize(opt)
    print("Test dataset contains %d input labels" % (len(dataloader_generate)))

    # model init
    from util.util import stop_grad
    from models.pix2pix_model import Pix2PixModel
    from util.visualizer import Visualizer

    model_generate = Pix2PixModel(opt)
    model_generate.eval()
    stop_grad(model_generate.netG)

    # util init
    visualizer = Visualizer(opt)

    # test and save results in temp dir (derived from opt.outputpath)
    tmp_image_dir = opt.output_path + '_tmp'
    if os.path.exists(tmp_image_dir):
        rmtree(tmp_image_dir)
    os.makedirs(tmp_image_dir)
    print("#############################")
    print("### create temp image dir ###")
    print("#############################")

    # iter over test dataset
    with jt.no_grad():
        for i, data_i in tqdm(enumerate(dataloader_generate)):
            if i * opt.batchSize >= opt.how_many:
                break
            generated = model_generate(data_i, mode='inference')[0]
            img_path = data_i['path']

            for b in range(generated.shape[0]):
                # print('process image... %s' % img_path[b])
                visuals = OrderedDict([('synthesized_image', generated[b])])
                visualizer.save_images_for_test(visuals, tmp_image_dir, img_path[b:b + 1], save_w=256, save_h=192)
    # clean
    jt.sync_all(True)
    jt.clean_graph()
    jt.gc()


def rescale_image(opt):
    # dir init
    tmp_image_dir_path = Path(opt.output_path + '_tmp')
    image_dir_path = Path(opt.output_path)
    if image_dir_path.exists():
        rmtree( opt.output_path )
    image_dir_path.mkdir()
    print("#############################")
    print("## create output image dir ##")
    print("#############################")

    if not opt.use_sr:
        ###############################
        # 直接使用PIL Image resize缩放 #
        ###############################
        for img_path in tqdm(tmp_image_dir_path.iterdir()):
            img = Image.open(img_path)
            scaled_img = img.resize((512,384), Image.BICUBIC)
            scaled_img.save( image_dir_path / img_path.name )
    else:
        ###############################
        # 使用jittor RCAN 超分辨率缩放 #
        ###############################
        from RCAN.models import RCAN
        from RCAN.dataset import SRTestDataset
        from RCAN.utils import stop_grad
        # sr model init
        model_sr = RCAN()
        state_dict = jt.load("./RCAN/best.pkl")
        model_sr.load_state_dict( state_dict["model"] )
        stop_grad(model_sr)
        # dataset init
        LR_folder = str(tmp_image_dir_path.absolute())
        sr_dataset = SRTestDataset(LR_folder)
        sr_dataset.set_attrs(batch_size=1, 
                             shuffle=False, 
                             drop_last=False, 
                             num_workers=int(opt.nThreads))
        # iter over sr dataset
        with jt.no_grad():
            for lr,filename in tqdm(sr_dataset):
                hr = model_sr(lr)
                hr = jt.clamp(hr, min_v=0.0, max_v=255.0).uint8()
                # iter over batch
                for i in range(hr.shape[0]):
                    # chw -> hwc
                    hr_pil = Image.fromarray( hr[i].numpy().transpose((1, 2, 0) ))
                    hr_pil.save( image_dir_path / filename[i] )
        # clean
        jt.sync_all(True)
        jt.clean_graph()
        jt.gc()
    
    # remove tmp dir
    rmtree( opt.output_path + '_tmp' )
    print("#############################")
    print("### remove temp image dir ###")
    print("#############################")


if __name__ == "__main__":
    # global config
    opt = TestOptions().parse()
    # 如果output的格式不为'./testB'，而是'./testB/'，先将末尾的'/'去掉
    if opt.output_path[-1] == '/':
        opt.output_path = osp.split(opt.output_path)[0]
    fix_seed(seed = 628)

    # jittor config
    jt.flags.use_cuda = (jt.has_cuda and opt.gpu_ids != "-1")

    # generate from label
    label_to_image(opt)

    # rescale results
    rescale_image(opt)

