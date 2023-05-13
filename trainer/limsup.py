#!coding:utf-8
import torch

import os
import datetime
from pathlib import Path
from itertools import cycle
from collections import defaultdict


class Trainer:

    def __init__(self, model, optimizer, device, config):
        print('limsup')
        self.model = model
        self.optimizer = optimizer
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.save_dir = '{}-{}_{}'.format(config.arch, config.model,
                                                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.save_dir = os.path.join(config.save_dir, self.save_dir)
        self.save_freq = config.save_freq
        self.print_freq = config.print_freq
        self.device = device
        self.epoch = 0

    def train_iteration(self, label_loader, print_freq):
        loop_info = defaultdict(list)
        batch_idx, label_n, unlab_n = 0, 0, 0
        for (label_x, label_y, ldx) in label_loader:
            label_x, label_y = label_x.to(self.device), label_y.to(self.device)
            lbs, ubs = label_x.size(0), -1
            batch_idx, label_n, unlab_n = batch_idx + 1, label_n + lbs, unlab_n + ubs
            ##=== forward ===
            _, outputs = self.model(label_x)
            loss = self.ce_loss(outputs, label_y)
            loop_info['lloss'].append(loss.item() * lbs)
            ## backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===

            loop_info['lacc'].append(label_y.eq(outputs.max(1)[1]).float().sum().item())
            # if print_freq > 0 and (batch_idx % print_freq) == 0:
            #     print(f"[train][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(">>>[train]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n, unlab_n

    def test_iteration(self, data_loader, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, (data, targets, ldx) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1

            ##=== forward ===
            _, outputs = self.model(data)
            loss = self.ce_loss(outputs, targets)
            loop_info['lloss'].append(loss.item())

            ##=== log info ===
            label_n, unlab_n = label_n + lbs, unlab_n + ubs
            loop_info['lacc'].append(targets.eq(outputs.max(1)[1]).float().sum().item())
            # if print_freq > 0 and (batch_idx % print_freq) == 0:
            #     print(f"[test][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[valid]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def train(self, label_loader, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            return self.train_iteration(label_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader, print_freq)

    def loop(self, epochs, label_data, test_data, scheduler=None):
        best_ep, best_acc = 0, 0.

        val_epoch = epochs * 0.8
        print('Validation epoch {}'.format(val_epoch))

        info = defaultdict(list)
        for ep in range(epochs):
            self.epoch = ep
            print("------ Training epochs: {} ------".format(ep+1))
            t_info, ln, un = self.train(label_data, self.print_freq)
            info['train_loss'].append(sum(t_info['lloss']) / ln)
            info['train_acc'].append(sum(t_info['lacc']) / ln)
            if scheduler is not None: scheduler.step()
            print("------ Validing epochs: {} ------".format(ep+1))
            e_info, e_n = self.test(test_data, self.print_freq)
            info['valid_loss'].append(sum(e_info['lloss']) / e_n)
            info['valid_acc'].append(sum(e_info['lacc']) / e_n)
            acc = sum(e_info['lacc']) / e_n

            # if acc > best_acc:
            #     best_ep, best_acc = ep, acc
            #     self.save()

            if self.epoch > val_epoch:
                if acc > best_acc:
                    best_ep, best_acc = ep, acc
                    self.save()

        print(">>>[best result]", best_ep, best_acc)
        torch.save(info, Path(self.save_dir)/"params.pkl")
        return Path(self.save_dir)

    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {'l': lbs, 'u': ubs, 'a': lbs + ubs}
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