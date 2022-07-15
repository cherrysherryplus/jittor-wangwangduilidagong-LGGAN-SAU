import jittor.nn as nn


class Loss(nn.Module):
    def __init__(self):
        self.l1_loss = nn.L1Loss()
        self.weight_l1 = 1.0

    def execute(self, sr, hr):
        return self.weight_l1 * self.l1_loss(sr, hr)

        