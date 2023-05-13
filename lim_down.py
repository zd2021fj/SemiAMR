#!coding:utf-8
from utils.datasets_v1 import Radio_data_v1
import pickle as pk
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from utils.ramps import exp_warmup
from utils.config import parse_commandline_args
from architectures.arch import arch
from utils.seed import setup_seed
from utils.test_tool import result_modelv1

import test
from trainer import limsup
build_model = {
    'limsup': limsup.Trainer,
}

def data_decode(dataset):
    # 加载数据路径
    # 加载label
    # snrs [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    # mods ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    # Xd = pk.load(open('../data/RML2016.10a_dict.pkl', 'rb'), encoding='iso-8859-1')
    # Xd = pk.load(open('/data1/ypg/data/RF data/2016.04C/2016.04C.multisnr/2016.04C.multisnr.pkl', 'rb'), encoding='latin')
    # Xd = pk.load(open('/data1/ypg/data/RF data/2016.10A/RML2016.10a/RML2016.10a_dict.pkl', 'rb'), encoding='iso-8859-1')
    # Xd = pk.load(open('../data/RML2016.10a_dict.pkl', 'rb'), encoding='iso-8859-1')
    # Xd = pk.load(open("/data1/ypg/data/RF data/2016.10A/RML2016.10b/RML2016.10b.dat", 'rb'), encoding='latin')
    if dataset == '10a':
        Xd = pk.load(open('/data1/ypg/data/RF data/2016.10A/RML2016.10a/RML2016.10a_dict.pkl', 'rb'),
                     encoding='iso-8859-1')
    elif dataset == '10b':
        Xd = pk.load(open("/data1/ypg/data/RF data/2016.10A/RML2016.10b/RML2016.10b.dat", 'rb'), encoding='latin')
    elif dataset == '04c':
        Xd = pk.load(open('/data1/ypg/data/RF data/2016.04C/2016.04C.multisnr/2016.04C.multisnr.pkl', 'rb'),
                     encoding='latin')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0]))
    lbl = []
    for mod in mods:
        for snr in snrs:
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    Label = list(map(lambda x: lbl[x][0], list(set(range(0, len(lbl))))))
    Labels = list(map(lambda x: mods.index(Label[x]), list(set(range(0, len(lbl))))))
    SNR = list(map(lambda x: lbl[x][1], list(set(range(0, len(lbl))))))
    return Labels, SNR, mods

def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps is None: return None
        if isinstance(config.steps, int): config.steps = [config.steps]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler


def run(config):
    print(config)
    print("pytorch version : {}".format(torch.__version__))

    print("model:{}\tlabel_rate:{}\tprogram seed:{}\tdata seed:{}\tepochs:{}\tsnr:{}\tlearning_rate:{}".format(
        config.model, config.label_rate, config.program_seed, config.data_seed, config.epochs,
        config.snr, config.lr))

    ## create save directory
    if config.save_freq !=0 and not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## prepare data
    setup_seed(config.program_seed)

    dir_list = {
        '10a': '/data1/ypg/data/RF data/2016.10A/TF/STFTN8',
        '10b': '/data1/ypg/data/RF data/2016.10b/TF/STFTN8',
       '04c': '/data1/ypg/data/RF data/2016.04C/TF/STFTN8'
    }

    # Xn = list(sorted(os.listdir(dirs2)))
    Labels, SNR, mods = data_decode(config.dataset_name)

    dirs1 = dir_list[config.program_seed]
    print('load data root: {}\n'.format(dirs1))
    Xm = list(sorted(os.listdir(dirs1)))

    ## sig_dataset
    sig_dataset = Radio_data_v1(dirs1, Xm, SNR, Labels, len(mods), config)

    ## Datalodar
    dataloaders = {x: torch.utils.data.DataLoader(sig_dataset[x], shuffle=True, num_workers=config.workers,
                                                  batch_size=config.lsp_batch_size)
                   for x in ['train', 'label','valid', 'test']
                   }

    ## prepare architecture
    net = arch[config.arch](sig_dataset['num_classes'], config.drop_ratio)
    net = nn.DataParallel(net)
    net = net.to(device)
    optimizer = create_optim(net.parameters(), config)
    scheduler = create_lr_scheduler(optimizer, config)

    ## run the model
    trainer = build_model[config.model](net, optimizer, device, config)

    ## 10%sup
    print("--------------------下届实验--------------------")
    dir = trainer.loop(config.epochs, dataloaders['label'], dataloaders['valid'], scheduler=scheduler)
    ## test the model
    test.tester(dir, result_modelv1, mods, dataloaders['test'], device, config.version)



if __name__ == '__main__':
    config = parse_commandline_args()
    run(config)