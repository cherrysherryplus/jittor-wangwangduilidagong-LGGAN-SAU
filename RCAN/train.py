# 【不选择超分辨率模型缩放生成图像时，RCAN不必再训练】
# 训练前，需要在原数据集上构造512*384和256*192两个版本的数据集；
# 训练时，在bash中，进入到RCAN/目录下，运行
# python train.py --input_path <数据集相对于RCAN/train.py的相对路径或绝对路径>

import os
import jittor as jt
from jittor.lr_scheduler import StepLR
import argparse
from tqdm import tqdm

from models import RCAN
from dataset import SRDataset
from loss import Loss
from utils import *
from tensorboardX import SummaryWriter


# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", 
                    type=str, 
                    default="../datasets/landscape", 
                    help="input_path设置为train、val、testA及testB所在的上级目录")
opt = parser.parse_args()

# seed cuda
jt.flags.use_cuda = 1
jt.set_global_seed(628)

# continue train
continue_train = False

# model
model = RCAN()

# optim
lr = 2e-4
optim = jt.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optim, step_size=25, gamma=0.5)

# loss
loss = Loss()

# data
DATA_folder = opt.input_path
BATCH_SIZE = 32
BATCH_SIZE_EVAL = 4
train_dataset = SRDataset(DATA_folder, train=True, augment=True, scale=2, colors=3, repeat=2, patch_size=96)
train_dataset.set_attrs(batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

val_dataset = SRDataset(DATA_folder, train=False, augment=True, scale=2, colors=3, repeat=1)
val_dataset.set_attrs(batch_size=BATCH_SIZE_EVAL, shuffle=True, drop_last=False, num_workers=2)

# writer
writer = SummaryWriter(logdir="./tf-log")

# start grad
start_grad(model)

# start train
START_EPOCH = 0
EPOCH = 200
BEST_PSNR = 0.0
EVAL_INTERVAL = 1
SAMPLE_INTERVAL = 500
SAMPLE_INTERVAL_EVAL = len(val_dataset) // 2

if continue_train:
    state_dict = None
    for filename in os.listdir("./"):
        if "latest" in filename:
            state_dict = jt.load(filename)
        if "best" in filename:
            # best psnr
            BEST_PSNR = float(os.path.splitext(filename)[0].split('_')[-1])
            print("BEST PSNR is %.3f" % BEST_PSNR)
    if state_dict is None:
        exit("no latest checkpoint to resume from")
    else:
        # start epoch
        START_EPOCH = state_dict["epoch"]
        print("resume from epoch ", START_EPOCH)
        START_EPOCH = START_EPOCH + 1
        # model
        model.load_state_dict(state_dict["model"])
        # optim
        optim.load_state_dict(state_dict["optim"])
        # scheduler
        scheduler.step_size = state_dict["scheduler"]["step_size"]
        scheduler.gamma = state_dict["scheduler"]["gamma"]
        scheduler.last_epoch = state_dict["scheduler"]["last_epoch"] + 1
        scheduler.cur_epoch = START_EPOCH + 1
    

for epoch in range(START_EPOCH, EPOCH):
    ##############################################
    # train
    ##############################################
    print(f"---------------epoch: {epoch}---------------")
    print("----------------------------------------------")
    for it,(hr,lr,filename) in tqdm(enumerate(train_dataset)):
        hr_ = model(lr)
        loss_v = loss(hr_, hr)
        writer.add_scalar("loss/train", loss_v.item(), it+epoch*len(train_dataset))
        optim.step(loss_v)
        if (it+1) % SAMPLE_INTERVAL == 0:
            with jt.no_grad():
                outp = jt.concat([hr_, hr], dim=2)
                outp = jt.clamp(outp, min_v=0.0, max_v=255.0) * (1.0 / 255.0)
                writer.add_images("sample/train", outp.numpy(), it+epoch*len(train_dataset))

    jt.sync_all(True)
    jt.clean_graph()
    jt.gc()

    ##############################################
    # eval
    ##############################################
    if (epoch+1) % EVAL_INTERVAL == 0:
        stop_grad(model)
        with jt.no_grad():
            psnr_total = 0.0
            loss_v_val = 0.0
            for it_eval,(hr,lr,filename) in tqdm(enumerate(val_dataset)):
                hr_ = model(lr)
                psnr_total += calc_psnr(hr_,hr)
                loss_v_val += loss(hr_, hr)
                if (it_eval+1) % SAMPLE_INTERVAL_EVAL == 0:
                    outp_eval = jt.concat([hr_, hr], dim=2)
                    outp_eval = jt.clamp(outp_eval, min_v=0.0, max_v=255.0) * (1.0 / 255.0)
                    writer.add_images("sample/val", outp_eval.numpy(), it_eval+epoch*len(val_dataset))  
            # psnr计算时多除了BATCH_SIZE_EVAL，所以真实psnr要乘上BATCH_SIZE_EVAL(4)
            psnr = psnr_total / 400.0
            writer.add_scalar("psnr", psnr, epoch)
            writer.add_scalar("loss/val", loss_v_val.item()* BATCH_SIZE_EVAL / 400.0, epoch)
            state_dict = {
                "model":model.state_dict(),
                "optim":optim.state_dict(),
                "scheduler":{
                    "cur_epoch":epoch,
                    "last_epoch":scheduler.last_epoch,
                    "step_size":scheduler.step_size,
                    "gamma":scheduler.gamma,
                },
                "epoch":epoch,
                "step":it,
                "psnr":psnr
            }
            jt.save(state_dict, "./latest.pkl")
            if psnr > BEST_PSNR:
                BEST_PSNR = psnr
                jt.save(state_dict, f"./best.pkl")
        start_grad(model)
    scheduler.step()
    jt.sync_all(True)
    jt.clean_graph()
    jt.gc()

