#!coding:utf-8
import torch.nn.functional as F
import torch
import torch.nn as nn

def kl_div_with_logit(input_logits, target_logits,reduction='mean'):
    assert input_logits.size()==target_logits.size()
    targets = F.softmax(target_logits, dim=1)
    return F.kl_div(F.log_softmax(input_logits,1), targets,reduction)

def entropy_y_x(logit):
    soft_logit = F.softmax(logit, dim=1)
    return -torch.mean(torch.sum(soft_logit* F.log_softmax(logit,dim=1), dim=1))

def softmax_loss_no_reduce(input_logits, target_logits, eps=1e-10):
    assert input_logits.size()==target_logits.size()
    target_soft = F.softmax(target_logits, dim=1)
    return -torch.sum(target_soft* F.log_softmax(input_logits+eps,dim=1), dim=1)

def softmax_loss_mean(input_logits, target_logits, eps=1e-10):
    assert input_logits.size()==target_logits.size()
    target_soft = F.softmax(target_logits, dim=1)
    return -torch.mean(torch.sum(target_soft* F.log_softmax(input_logits+eps,dim=1), dim=1))

def sym_mse(logit1, logit2):
    assert logit1.size()==logit2.size()
    return torch.mean((logit1 - logit2)**2)

def sym_mse_with_softmax(logit1, logit2):
    assert logit1.size()==logit2.size()
    return torch.mean((F.softmax(logit1,1) - F.softmax(logit2,1))**2)

def mse_with_softmax(logit1, logit2,reduction='mean'):
    assert logit1.size()==logit2.size()
    return F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1),reduction)

def one_hot(targets, nClass):
    logits = torch.zeros(targets.size(0), nClass).to(targets.device)
    return logits.scatter_(1,targets.unsqueeze(1),1)

def label_smooth(one_hot_labels, epsilon=0.1):
    nClass = one_hot_labels.size(1)
    return ((1.-epsilon)*one_hot_labels + (epsilon/nClass))

def uniform_prior_loss(logits):
    logit_avg = torch.mean(F.softmax(logits,dim=1), dim=0)
    num_classes, device = logits.size(1), logits.device
    p = torch.ones(num_classes).to(device) / num_classes
    return -torch.sum(torch.log(logit_avg) * p)

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
