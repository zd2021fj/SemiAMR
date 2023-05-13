#!coding:utf-8
import torch
from torch.nn import functional as F
import os
import datetime
from pathlib import Path
from collections import defaultdict
from itertools import cycle
from utils.mixup import *
from utils.fmix import Fmix_double
from utils.ramps import exp_rampup
from utils.loss import one_hot

class Trainer:

    def __init__(self, model, ema_model,optimizer, device, config):
        print("SemiAMR two fmix mt model fmix mask(model - ema) mse(mix model - mix ema output)")
        self.model = model
        self.ema_model = ema_model
        self.optimizer  = optimizer
        self.lce_loss   = torch.nn.CrossEntropyLoss()
        self.mixup_loss = mixup_ce_loss_with_softmax2 # mixup_mse_loss_with_softmax
        self.save_dir  = '{}-{}_{}'.format(config.arch, config.model,
                          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.save_dir  = os.path.join(config.save_dir, self.save_dir)
        self.usp_weight  = config.usp_weight
        self.alpha = config.fmix_alpha
        self.save_freq   = config.save_freq
        self.print_freq  = config.print_freq
        self.device      = device
        self.epoch       = 0
        self.global_step = 0
        self.ema_decay = config.ema_decay
        self.rampup = exp_rampup(config.weight_rampup)

    def train_iteration(self, label_loader, unlab_loader, print_freq):
        loop_info = defaultdict(list)
        batch_idx, label_n, unlab_n = 0, 0, 0
        for (label, label_target, ldx), (unlabel, unlabel_target, udx) in zip(cycle(label_loader), unlab_loader):

            label, label_target = label.to(self.device), label_target.to(self.device)
            unlabel, unlabel_target = unlabel.to(self.device), unlabel_target.to(self.device)

            lbs, ubs = label.size(0), unlabel.size(0)
            batch_idx, label_n, unlab_n = batch_idx + 1, label_n + lbs, unlab_n + ubs

            _, outputs = self.model(label)
            lloss = self.lce_loss(outputs, label_target)
            loop_info['lloss'].append(lloss.item() * lbs)

            # 获取epoch伪标签
            iter_unlab_pslab = self.epoch_pslab[udx]
            # 获取单个混合后的mix
            # 因为使用unlabel得到的mix，所以伪标签应该用原来的数据更新
            mixed_u1, mixed_u2, uy_a, uy_b, lam = Fmix_double(unlabel, iter_unlab_pslab, self.alpha, self.device)

            # 获取mix1的分类预测值
            _, mixed_output_u1 = self.model(mixed_u1)

            # 不更新梯度
            with torch.no_grad():
                # 获取无标签数据的分类预测值
                _, unlab_preds = self.model(unlabel)
                pseudo_preds = unlab_preds.clone()
                # # 使用无标签model的预测值更新伪标签的值
                self.epoch_pslab[udx] = pseudo_preds.detach()

                _, ema_unlab_preds = self.ema_model(unlabel)
                ema_unlab_preds = ema_unlab_preds.detach()

                # 计算ema模型mix2的预测值
                _,mixed_ema_preds_u2 = self.ema_model(mixed_u2)
                mixed_ema_preds_u2 = mixed_ema_preds_u2.detach() # ema模型梯度不更新

            # 计算mask
            mse = torch.sum(F.mse_loss(F.softmax(mixed_output_u1, 1), F.softmax(mixed_ema_preds_u2, dim=1), reduction='none'), dim=1)
            # 计算model mix1 的mix ce loss
            uloss = torch.mean(torch.exp(-mse) * self.mixup_loss(mixed_output_u1, uy_a, uy_b,lam)) + torch.mean(mse)
            uloss =  uloss * self.rampup(self.epoch) * self.usp_weight
            loop_info['uloss'].append(uloss.item() * ubs)

            loss = lloss + uloss
            loop_info['aloss'].append(loss.item() * (lbs + ubs))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新ema模型的参数
            self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)

            loop_info['lacc'].append(label_target.eq(outputs.max(1)[1]).float().sum().item())
            loop_info['uacc'].append(unlabel_target.eq(unlab_preds.max(1)[1]).float().sum().item())

        print(f">>>[train]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n, unlab_n

    def test_iteration(self, data_loader, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, (x1,targets,ldx) in enumerate(data_loader):
            data, targets = x1.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1

            _, outputs = self.model(data)
            loss = self.lce_loss(outputs, targets)
            loop_info['lloss'].append(loss.item() * lbs)

            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(targets.eq(outputs.max(1)[1]).float().sum().item())
            # if print_freq>0 and (batch_idx%print_freq)==0:
            #     print(f"[test][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[valid]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def train(self, label_loader, unlab_loader, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            return self.train_iteration(label_loader, unlab_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader, print_freq)

    def loop(self, epochs, sig_dataset, label_data, unlab_data, test_data, scheduler=None):

        self.epoch_pslab = self.create_soft_pslab(n_samples=len(sig_dataset['unlabel']),
                                                  n_classes=sig_dataset['num_classes'])
        best_ep, best_acc = 0, 0.
        val_epoch = epochs * 0.8
        print('Validation epoch {}'.format(val_epoch))
        info = defaultdict(list)
        for ep in range(epochs):
            self.epoch = ep
            print("------ Training epochs: {} ------".format(ep + 1))
            t_info, ln, un = self.train(label_data, unlab_data, self.print_freq)
            info['train_lloss'].append(sum(t_info['lloss']) / ln)
            info['train_klloss'].append(sum(t_info['klloss']) / un)
            info['train_uloss'].append(sum(t_info['uloss']) / un)
            info['train_loss'].append(sum(t_info['aloss']) / (ln + un))
            info['train_lacc'].append(sum(t_info['lacc']) / ln)
            info['train_uacc'].append(sum(t_info['uacc']) / un)
            info['train_acc'].append(sum(t_info['lacc']) / ln + sum(t_info['uacc']) / un)
            if scheduler is not None: scheduler.step()
            print("------ Validing epochs: {} ------".format(ep + 1))
            e_info, e_n = self.test(test_data, self.print_freq)
            info['valid_loss'].append(sum(e_info['lloss']) / e_n)
            info['valid_acc'].append(sum(e_info['lacc']) / e_n)
            acc = sum(e_info['lacc']) / e_n

            if self.epoch > val_epoch:
                if acc > best_acc:
                    best_ep, best_acc = ep, acc
                    self.save()
        print(">>>[best result]", best_ep, best_acc)
        torch.save(info, Path(self.save_dir) / "params.pkl")
        return Path(self.save_dir)

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_((param.data)*(1-alpha))

    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {'l': lbs, 'u': ubs, 'k': lbs, 'a': lbs + ubs}
        for k, val in info.items():
            n = nums[k[0]]
            v = val[-1] if iteration else sum(val)
            s = f'{k}: {v / n:.3%}' if k[-1] == 'c' else f'{k}: {v / n:.5f}'
            ret.append(s)
        return '\t'.join(ret)

    def save(self, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            # state = self.model.state_dict()
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_target = model_out_path / "model.pth"
            ema_save_dir = model_out_path / "ema_model.pth"
            torch.save(self.model, save_target)
            torch.save(self.ema_model, ema_save_dir)

    def create_soft_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype=='rand':
            rlabel = torch.randint(0, n_classes, (n_samples,)).long()
            pslab  = one_hot(rlabel, n_classes)
        elif dtype=='zero':
            pslab = torch.zeros(n_samples, n_classes)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        return pslab.to(self.device)#!coding:utf-8

    def save_checkpoints(self, scheduler):
        state = {'model':self.model.stae_dict(),
                 'ema_model': self.ema_model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'lr_scheduler': scheduler,
                 'epoch': self.epoch}

        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            # state = self.model.state_dict()
            if not model_out_path.exists():
                model_out_path.mkdir()

            save_target = os.path.join(model_out_path, '%s_%s' % ("checkpoint", self.epoch))

            torch.save(state, save_target)
