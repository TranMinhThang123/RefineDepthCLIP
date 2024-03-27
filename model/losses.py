# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        return loss

class L1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,pred,target):
        abs_diff = torch.abs(target-pred)
        loss = torch.mean(abs_diff)
        return loss
    
class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_loss = SiLogLoss()
        self.l1_loss = L1Loss()
    def forward(self,pred,target):
        loss = self.log_loss(pred,target)+self.l1_loss(pred,target)
        return loss