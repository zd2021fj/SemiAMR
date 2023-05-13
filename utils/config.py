import argparse

from numpy.distutils.fcompiler import str2bool


def create_parser():
    parser = argparse.ArgumentParser(description='Semi supevised Training --PyTorch ')  # 生成参数解析器
    # 添加信息
    # Log and save
    parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='display frequence (default: 20)')
    parser.add_argument('--save_freq', default=100, type=int, metavar='EPOCHS', help='checkpoint frequency(default: 0)')
    parser.add_argument('--save_dir', default='./checkpoints/ours/10a', type=str, metavar='DIR')

    # Data
    parser.add_argument('--dataset_name', default='10a', type=str, help='10a 10b 04c')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--snr', default=0, type=int, metavar='N',
                        help='snr (default: 0)')
    parser.add_argument('--lsp_batch_size', default=64, type=int, metavar='N',
                        help=' label batch size for data (default: 64)')
    parser.add_argument('--usp_batch_size', default=64, type=int, metavar='N',
                        help='unlabel batch size for data (default: 64)')
    parser.add_argument('--data_seed', default=0, type=int, metavar='N',
                        help='number of data seed (default: 0)')
    parser.add_argument('--program_seed', default=0, type=int, metavar='N',
                        help='number of progarm seed (default: 0)')
    parser.add_argument('--label_rate', default=0.01, type=float, metavar='N',
                        help='label-rate(default: 0.01)')
    # Architecture
    parser.add_argument('--version', default=1, type=int,
                        help='model output one/two/three')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='sslnet')
    parser.add_argument('--model', metavar='MODEL', default='SemiAMR')
    parser.add_argument('--drop_ratio', default=0.5, type=float, help='ratio of dropout (default: 0)')

    # Optimization
    parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of total training epochs')
    parser.add_argument('--optim', default='adam', type=str, metavar='TYPE', choices=['sgd', 'adam'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', default=False, type=bool, metavar='BOOL',
                        help='use nesterov momentum (default: False)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

    # LR schecular
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                        help='max learning rate (default: 0.1)')
    parser.add_argument('--lr_scheduler', default="cos", type=str, choices=['cos', 'multistep', 'exp-warmup', 'none'])
    parser.add_argument('--min_lr', default=1e-8, type=float, metavar='LR',
                        help='minimum learning rate (default: 1e-4)')
    parser.add_argument('--steps', type=int, nargs='+', metavar='N', help='decay steps for multistep scheduler')
    parser.add_argument('--gamma', type=float, help='factor of learning rate decay')
    parser.add_argument('--rampup_length', type=int, metavar='EPOCHS', help='length of the ramp-up')
    parser.add_argument('--rampdown_length', type=int, metavar='EPOCHS', help='length of the ramp-down')

    # Pseudo-Label 2013
    parser.add_argument('--t1', type=float, metavar='EPOCHS', help='T1')
    parser.add_argument('--t2', type=float, metavar='EPOCHS', help='T1')
    parser.add_argument('--soft', type=str2bool, help='use soft pseudo label')

    # VAT
    parser.add_argument('--xi', type=float, metavar='W', help='xi for VAT')
    parser.add_argument('--eps', type=float, metavar='W', help='epsilon for VAT')
    parser.add_argument('--n_power', type=int, metavar='N',
                        help='the iteration number of power iteration method in VAT')

    # Fixmatch
    parser.add_argument('--threshold', type=float, metavar='W', help='threshold for confident predictions in Fixmatch')

    # MeanTeacher-based method
    parser.add_argument('--ema_decay', type=float, default=0.99, metavar='W', help='ema weight decay')

    # Mixup-based method
    parser.add_argument('--mixup_alpha', default=1.0, type=float, metavar='W', help='mixup alpha for beta distribution')
    parser.add_argument('--cutmix_alpha', type=float, metavar='W', help='mixup alpha for beta distribution')
    parser.add_argument('--fmix_alpha', type=float, default=1, metavar='W', help='mixup alpha for beta distribution')

    # Opt for loss
    parser.add_argument('--usp_weight', default=30.0, type=float, metavar='W',
                        help='the upper of unsuperivsed weight (default: 1.0)')
    parser.add_argument('--ukl_weight', default=1.0, type=float, metavar='W',
                        help='the upper of unsuperivsed weight (default: 1.0)')
    parser.add_argument('--weight_rampup', default=30, type=int, metavar='EPOCHS',
                        help='the length of rampup weight (default: 30)')
    parser.add_argument('--ent_weight', type=float, metavar='W', help='the weight of minEnt regularization')

    # SSR
    parser.add_argument('--ct_weight', type=float, default=0.003)
    parser.add_argument('--ct_lr', type=float, default=0.5)
    parser.add_argument('--kl_weight', type=float, default=10.0)


    return parser

def parse_commandline_args():
    return create_parser().parse_args()  # 以key-value形式存入arg中