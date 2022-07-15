# https://zhuanlan.zhihu.com/p/275742390
import os
import PIL.Image as Image
from tqdm import tqdm

import jittor as jt
import jittor.transform as TF
from jittor.dataset import *


def getStat(train_loader):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    mean = jt.zeros(3)
    std = jt.zeros(3)
    total = train_loader.total_len
    for X in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    return list(mean.divide(total).numpy()), list(std.divide(total).numpy())

class TestDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_names = os.listdir(self.root)
        self.set_attrs(total_len = len(self.img_names))
    def __getitem__(self, idx):
        img_path = os.path.join( self.root, self.img_names[idx] )
        img = Image.open(img_path)
        if self.transform:
            return self.transform(img)
        return img
 
if __name__ == '__main__':
    train_loader = TestDataset(root=r'./datasets/landscape/train/imgs_lr', transform=TF.ToTensor())
    train_loader.set_attrs(batch_size=1, shuffle=False, num_workers=0)
    print(len(train_loader))

    # output
    # [0.44166297, 0.49118593, 0.5296761], [0.22782151, 0.22858785, 0.25046638]
    print(getStat(train_loader))

    