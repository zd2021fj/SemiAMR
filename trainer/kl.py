#!coding:utf-8
import torch
from torch.nn import functional as F

import os
import datetime
from pathlib import Path
from collections import defaultdict
from itertools import cycle

from utils.loss import kl_div_with_logit
from utils.mixup import *

class Trainer:

    def __init__(self, model, optimizer, device, config):
        print("KL")
        self.model      = model
        self.optimizer  = optimizer
        self.lce_loss   = torch.nn.CrossEntropyLoss()
        self.kl_loss    = kl_div_with_logit
        self.save_dir  = '{}-{}_{}'.format(config.arch, config.model,
                          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.save_dir  = os.path.join(config.save_dir, self.save_dir)
        self.ukl_weight = config.ukl_weight
        self.save_freq   = config.save_freq
        self.print_freq  = config.print_freq
        self.device      = device
        self.epoch       = 0

    def train_iteration(self, label_loader, unlab_loader, print_freq):
        loop_info = defaultdict(list)
        batch_idx, label_n, unlab_n = 0, 0, 0
        for (x1,_, label_y,ldx), (u1,u2, unlab_y,udx) in zip(cycle(label_loader), unlab_loader):
            batch_idx+=1;
            label_x, unlab_u1, unlab_u2 = x1.to(self.device), u1.to(self.device), u2.to(self.device)
            label_y, unlab_y = label_y.to(self.device), unlab_y.to(self.device)
            lbs, ubs = x1.size(0), u2.size(0)

            ##=== forward ===
            outputs = self.model(label_x)
            lloss = self.lce_loss(outputs, label_y)
            loop_info['lloss'].append(lloss.item()*lbs)

            ##=== Semi-supervised Training ===
            ## kl loss for unlabeled data
            unlab_outputs1 = self.model(unlab_u1)
            unlab_outputs2 = self.model(unlab_u2)
            klloss = self.kl_loss(unlab_outputs1,unlab_outputs2)
            klloss *= self.ukl_weight
            loop_info['klloss'].append(klloss.item() * ubs)

            ## cross-entropy loss for confident unlabeled data
            # iter_unlab_pslab = self.epoch_pslab[udx]
            # uloss = self.uce_loss(unlab_outputs1, iter_unlab_pslab)
            # uloss *= torch.exp(-klloss)*self.usp_weight
            # loop_info['uloss'].append(uloss.item() * ubs)
            ###loss
            loss = lloss+klloss;
            loop_info['aloss'].append(loss.item() * (lbs + ubs))
            ## use the outputs of weak unlabeled data as pseudo labels

            # with torch.no_grad():
            #     pseudo_preds = unlab_outputs1.max(1)[1]
            #     self.epoch_pslab[udx] = pseudo_preds.detach()

            ##=== backwark ===
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===
            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(label_y.eq(outputs.max(1)[1]).float().sum().item())
            loop_info['uacc'].append(unlab_y.eq(unlab_outputs1.max(1)[1]).float().sum().item())
            # if print_freq>0 and (batch_idx%print_freq)==0:
            #     print(f"[train][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[train]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n, unlab_n

    def test_iteration(self, data_loader, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, (x1,_,targets,ldx) in enumerate(data_loader):
            data, targets = x1.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1

            ##=== forward ===
            outputs = self.model(data)
            loss = self.lce_loss(outputs, targets)
            loop_info['lloss'].append(loss.item() * lbs)

            ##=== log info ===
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
        ## construct epoch pseudo labels
        # self.epoch_pslab, self.epoch_probs = self.create_pslab(
        #     n_samples=len(sig_dataset['unlabel']),
        #     n_classes=sig_dataset['num_classes'])
        ## main process
        best_ep, best_acc = 0, 0.
        info = defaultdict(list)
        for ep in range(epochs):
            self.epoch = ep
            print("------ Training epochs: {} ------".format(ep + 1))
            t_info, ln, un = self.train(label_data, unlab_data, self.print_freq)
            info['train_lloss'].append(sum(t_info['lloss']) / ln)
            info['train_klloss'].append(sum(t_info['klloss']) / un)
            # info['train_uloss'].append(sum(t_info['uloss']) / un)
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
            if acc > best_acc:
                best_ep, best_acc = ep, acc
                self.save()
        print(">>>[best result]", best_ep, best_acc)
        torch.save(info, Path(self.save_dir) / "params.pkl")
        return Path(self.save_dir)

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
            torch.save(self.model, save_target)

    def create_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype=='rand':
            pslab = torch.randint(0, n_classes, (n_samples,))
        elif dtype=='zero':
            pslab = torch.zeros(n_samples)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        probs = torch.ones(n_samples)
        return pslab.long().to(self.device), probs.to(self.device)