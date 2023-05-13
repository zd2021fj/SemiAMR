import torch
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import h5py
from torch.utils.data import Dataset

#Dataset for stft
class Dataset_v1(Dataset):
    """
    Return single input and targets
    """
    def __init__(self, Xa, Y, dirs):
        self.signalxa = Xa
        self.label = torch.Tensor(Y).long()
        self.dirs = dirs
    def __getitem__(self,index):
        xa = self.signalxa[index]
        xa = h5py.File(os.path.join(self.dirs, xa))
        xa = abs(xa['stft'][:])
        xa = torch.Tensor(xa/np.max(xa)).unsqueeze(0)
        y = self.label[index]
        return xa, y, index
    def __len__(self):
        return len(self.signalxa)

#split dataset
def split_dataset_v1(X_data, Y_data, random_state, n_splits=1, train_size=0.9):
    """
    Return split dataset accroding train_size
    """
    sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state = random_state)
    for train_index, test_index in sss.split(X_data, Y_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        Y_train, Y_test = Y_data[train_index], Y_data[test_index]
    return X_train, Y_train, X_test, Y_test

def Radio_data_v1(dirs, Xm, SNR, Labels, num_classes, config):
    """
        Return datasets from signal inputs and targets
    """
    ###加载数据集
    print('SNR {} '.format(config.snr))
    data_index = np.where(np.array(SNR) == config.snr)
    Xa_data = np.array(Xm)[data_index]
    Y_data = np.array(Labels)[data_index]
    num_classes = num_classes
    Xa_train, Y_train, Xa_test, Y_test = split_dataset_v1(Xa_data, Y_data, n_splits=1, train_size=0.9,
                                                       random_state=config.data_seed)  ###train 80% test20%
    Xa_tra, Y_tra, Xa_val, Y_val = split_dataset_v1(Xa_train, Y_train, n_splits=1, train_size=0.9,
                                                 random_state=config.data_seed)  ###train 72% valid 8%
    label_input, label_target, unlabel_input, unlabel_target = split_dataset_v1(Xa_tra, Y_tra, n_splits=1,
                                                                             train_size=config.label_rate,
                                                                             random_state=config.data_seed)

    print('总{}\t验证{}\t训练{}\t测试{}\t有标签{}\t无标签{}'.format(len(Xa_data), len(Xa_val), len(Xa_tra), len(Xa_test),
                                                      len(label_input), len(unlabel_input)))

    # Dataset
    sig_dataset = {'train': Dataset_v1(Xa_tra, Y_tra, dirs),
                   'label': Dataset_v1(label_input, label_target, dirs),
                   'unlabel': Dataset_v1(unlabel_input, unlabel_target, dirs),
                   'valid': Dataset_v1(Xa_val, Y_val, dirs),
                   'test': Dataset_v1(Xa_test, Y_test, dirs),
                   'num_classes': num_classes
                   }
    return sig_dataset


def Radio_04cdata_v1(dirs, Xm, SNR, Labels, num_classes, config):
    """
        Return datasets from signal inputs and targets
    """
    ###加载数据集
    print('SNR {} '.format(config.snr))
    data_index = np.where(np.array(SNR) == config.snr)
    Xa_data = np.array(Xm)[data_index]
    Y_data = np.array(Labels)[data_index]
    num_classes = num_classes

    # X, Y, _, _ = split_dataset_v1(Xa_data, Y_data, n_splits=1, train_size=0.9,
    #                                                       random_state=config.data_seed)  ###train 80% test20%


    Xa_train, Y_train, Xa_test, Y_test = split_dataset_v1(Xa_data, Y_data, n_splits=1, train_size=0.9,
                                                       random_state=config.data_seed)  ###train 80% test20%

    # Xa_tra, Y_tra, Xa_val, Y_val = split_dataset_v1(Xa_train, Y_train, n_splits=1, train_size=0.9,
    #                                              random_state=config.data_seed)  ###train 72% valid 8%

    label_input, label_target, unlabel_input, unlabel_target = split_dataset_v1(Xa_train, Y_train, n_splits=1,
                                                                             train_size=config.label_rate,
                                                                             random_state=config.data_seed)

    print('总{}\t取{}\t训练{}\t测试{}\t有标签{}\t无标签{}'.format(len(Xa_data), len(Xa_data), len(Xa_train), len(Xa_test), len(label_input), len(unlabel_input)))
    # Dataset
    sig_dataset = {'train': Dataset_v1(Xa_train, Y_train, dirs),
                   'label': Dataset_v1(label_input, label_target, dirs),
                   'unlabel': Dataset_v1(unlabel_input, unlabel_target, dirs),
                   # 'valid': Dataset_v1(Xa_val, Y_val, dirs),
                   'test': Dataset_v1(Xa_test, Y_test, dirs),
                   'num_classes': num_classes
                   }
    return sig_dataset

def Radio_10adata_v1(dirs, Xm, SNR, Labels, num_classes, config):
    """
        Return datasets from signal inputs and targets
    """
    ###加载数据集
    print('SNR {} '.format(config.snr))
    data_index = np.where(np.array(SNR) == config.snr)
    Xa_data = np.array(Xm)[data_index]
    Y_data = np.array(Labels)[data_index]
    num_classes = num_classes

    X, Y, _, _ = split_dataset_v1(Xa_data, Y_data, n_splits=1, train_size=0.7364,
                                                          random_state=config.data_seed)  ###train 80% test20%

    Xa_train, Y_train, Xa_test, Y_test = split_dataset_v1(X, Y, n_splits=1, train_size=0.9,
                                                       random_state=config.data_seed)  ###train 80% test20%

    # Xa_tra, Y_tra, Xa_val, Y_val = split_dataset_v1(Xa_train, Y_train, n_splits=1, train_size=0.9,
    #                                              random_state=config.data_seed)  ###train 72% valid 8%

    label_input, label_target, unlabel_input, unlabel_target = split_dataset_v1(Xa_train, Y_train, n_splits=1,
                                                                             train_size=config.label_rate,
                                                                             random_state=config.data_seed)

    print('总{}\t取{}\t训练{}\t测试{}\t有标签{}\t无标签{}'.format(len(Xa_data), len(X), len(Xa_train), len(Xa_test), len(label_input), len(unlabel_input)))
    # Dataset
    sig_dataset = {'train': Dataset_v1(Xa_train, Y_train, dirs),
                   'label': Dataset_v1(label_input, label_target, dirs),
                   'unlabel': Dataset_v1(unlabel_input, unlabel_target, dirs),
                   # 'valid': Dataset_v1(Xa_val, Y_val, dirs),
                   'test': Dataset_v1(Xa_test, Y_test, dirs),
                   'num_classes': num_classes
                   }
    return sig_dataset

def Radio_10bdata_v1(dirs, Xm, SNR, Labels, num_classes, config):
    """
        Return datasets from signal inputs and targets
    """
    ###加载数据集
    print('SNR {} '.format(config.snr))
    data_index = np.where(np.array(SNR) == config.snr)
    Xa_data = np.array(Xm)[data_index]
    Y_data = np.array(Labels)[data_index]
    num_classes = num_classes

    X, Y, _, _ = split_dataset_v1(Xa_data, Y_data, n_splits=1, train_size=0.135,
                                                          random_state=config.data_seed)  ###train 80% test20%

    Xa_train, Y_train, Xa_test, Y_test = split_dataset_v1(X, Y, n_splits=1, train_size=0.9,
                                                       random_state=config.data_seed)  ###train 80% test20%

    # Xa_tra, Y_tra, Xa_val, Y_val = split_dataset_v1(Xa_train, Y_train, n_splits=1, train_size=0.9,
    #                                              random_state=config.data_seed)  ###train 72% valid 8%

    label_input, label_target, unlabel_input, unlabel_target = split_dataset_v1(Xa_train, Y_train, n_splits=1,
                                                                             train_size=config.label_rate,
                                                                             random_state=config.data_seed)

    print('总{}\t取{}\t训练{}\t测试{}\t有标签{}\t无标签{}'.format(len(Xa_data), len(X), len(Xa_train), len(Xa_test), len(label_input), len(unlabel_input)))
    # Dataset
    sig_dataset = {'train': Dataset_v1(Xa_train, Y_train, dirs),
                   'label': Dataset_v1(label_input, label_target, dirs),
                   'unlabel': Dataset_v1(unlabel_input, unlabel_target, dirs),
                   # 'valid': Dataset_v1(Xa_val, Y_val, dirs),
                   'test': Dataset_v1(Xa_test, Y_test, dirs),
                   'num_classes': num_classes
                   }
    return sig_dataset