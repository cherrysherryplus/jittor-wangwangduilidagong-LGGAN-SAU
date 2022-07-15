import jittor as jt
import math

def fix_seed(seed=628):
    jt.set_global_seed(seed)

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)
    return float(psnr)

def start_grad(model):
    for name, param in model.named_parameters():
        # meanshift
        if "sub_mean" in name or "add_mean" in name:
            print("skip meanshift layer")
            continue
        # batch norm
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

