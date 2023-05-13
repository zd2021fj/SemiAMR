#!coding:utf-8
import torch
from torch.nn import functional as F

import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2

def cutmix(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns cutmixed inputs, pairs of targets, and lambda
        """
    """1.设定lamda的值，服从beta分布"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1 - lam)
    """2.找到两个随机样本"""
    index = torch.randperm(x.size(0)).to(device)
    """3.生成剪裁区域B"""
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    """4.将原有的样本A中的B区域，替换成样本B中的B区域"""
    # 打乱顺序后的batch组和原有的batch组进行替换[对应id下]
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    # adjust lambda to exactly match pixel ratio
    """5.根据剪裁区域坐标框的值调整lam的值"""
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def mixup_one_target(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam*x + (1-lam)*x[index, :]
    mixed_y = lam*y + (1-lam)*y[index]
    return mixed_x, mixed_y, lam


def mixup_two_targets(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam*x + (1-lam)*x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_ce_loss_soft(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss for soft labels
    """
    mixup_loss_a = -torch.mean(torch.sum(targets_a* F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(targets_b* F.log_softmax(preds, dim=1), dim=1))

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss


def mixup_ce_loss_hard(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss
    """
    mixup_loss_a = F.nll_loss(F.log_softmax(preds, dim=1), targets_a)
    mixup_loss_b = F.nll_loss(F.log_softmax(preds, dim=1), targets_b)

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss


def mixup_ce_loss_with_softmax(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss
    """
    mixup_loss_a = -torch.mean(torch.sum(F.softmax(targets_a,1)* F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(F.softmax(targets_b,1)* F.log_softmax(preds, dim=1), dim=1))

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss

def mixup_ce_loss_with_softmax2(preds, targets_a, targets_b, lam):
    """ mixed categorical cross-entropy loss
    """
    mixup_loss_a = -torch.sum(F.softmax(targets_a,1)* F.log_softmax(preds, dim=1), dim=1)
    mixup_loss_b = -torch.sum(F.softmax(targets_b,1)* F.log_softmax(preds, dim=1), dim=1)

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss

def mixup_mse_loss_with_softmax(preds, targets_a, targets_b, lam,reduction='mean'):
    """ mixed categorical mse loss
    """
    mixup_loss_a = F.mse_loss(F.softmax(preds,1), F.softmax(targets_a,1),reduction)
    mixup_loss_b = F.mse_loss(F.softmax(preds,1), F.softmax(targets_b,1),reduction)

    mixup_loss = lam* mixup_loss_a + (1- lam)* mixup_loss_b
    return mixup_loss
