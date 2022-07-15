import jittor as jt
from jittor.dataset import Dataset
from jittor import transform as transforms

import os
import random
import numpy as np
import PIL.Image as Image
import time


def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = jt.float32(ndarray_chw)
    return tensor

def crop_patch(lr, hr, patch_size, scale, augment=True):
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    hp = patch_size
    lp = patch_size // scale
    lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
    hx, hy = lx * scale, ly * scale
    lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90: lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0,2)
    # numpy to tensor
    lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return lr_patch, hr_patch


class SRDataset(Dataset):
    def __init__(
        self, DATA_folder,
        train=True, augment=True, scale=2, colors=3, repeat=14,
        patch_size=96
    ):
        super().__init__()
        self.augment   = augment
        self.img_postfix = '.jpg'
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size
        self.repeat = repeat
        self.train = train
        if self.train:
            self.HR_folder = os.path.join(DATA_folder, "train", "imgs_lr")
            self.LR_folder = os.path.join(DATA_folder, "train", "imgs_lr_256")
        else:
            self.HR_folder = os.path.join(DATA_folder, "val", "imgs_lr")
            self.LR_folder = os.path.join(DATA_folder, "val", "imgs_lr_256")

        self.hr_filenames = os.listdir(self.HR_folder)
        self.lr_filenames = os.listdir(self.LR_folder)
        self.nums_trainset = len(self.hr_filenames)
        
        self.set_attrs(total_len = self.nums_trainset * self.repeat)

    def __getitem__(self, idx):
        idx = idx % self.nums_trainset
        filename = self.hr_filenames[idx]
        hr = np.array(Image.open(os.path.join( self.HR_folder, filename ))).astype(np.uint8)
        lr = np.array(Image.open(os.path.join( self.LR_folder, filename ))).astype(np.uint8)
        
        if self.train:
            train_lr_patch, train_hr_patch = crop_patch(lr, hr, self.patch_size, self.scale, True)
            return train_hr_patch, train_lr_patch, filename
        
        return ndarray2tensor(hr), ndarray2tensor(lr), filename


class SRTestDataset(Dataset):
    def __init__(self, LR_folder):
        super().__init__()
        self.LR_folder = LR_folder
        self.lr_filenames = os.listdir(self.LR_folder)
        self.nums_testset = len(self.lr_filenames)
        
        self.set_attrs(total_len = self.nums_testset)

    def __getitem__(self, idx):
        filename = self.lr_filenames[idx]
        lr = np.array(Image.open(os.path.join( self.LR_folder, filename ))).astype(np.uint8)
        return ndarray2tensor(lr), filename


if __name__ == "__main__":
    LR_folder = '/root/autodl-tmp/datasets/landscape/val/imgs_lr_256'
    testset = SRTestDataset(LR_folder)
    testset.set_attrs(batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    # 直接testset[idx]返回的数据会少一个维度，即batch维度
    for i,(lr,filename) in enumerate(testset):
        print(lr.shape, filename)
        if (i+1) % 5 == 0:
            break
    # for debug
    exit(0)

    DATA_folder = '/root/autodl-tmp/datasets/landscape'
    div2k = SRDataset(DATA_folder, augment=True, scale=2, colors=3, repeat=14, patch_size=96)

    print("numner of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        tlr, thr, filename = div2k[idx]
        print(tlr.shape, thr.shape, filename)
    end = time.time()
    print(end - start)