import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import os
import joblib
from pylab import *
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ion()   # interactive mode交互式模式
# get_ipython().magic('matplotlib inline')

# result_modelv1
def result_modelv1(model, dataloader, device, version=0):
    model.eval()
    label_list = []
    preds_list = []
    label_arr = []
    preds_arr = []
    testing_acc = 0.0
    length = 0
    with torch.no_grad():
        for batch_idx, (data, targets, ldx) in enumerate(dataloader):
            length = length + len(data)
            data, targets = data.to(device), targets.to(device)
            if version == 0:
                outputs = model(data)
            elif version == 1:
                _,outputs = model(data)
            elif version == 2:
                _, _,outputs = model(data)
            elif version == 3:
                outputs, _ = model(data)

            label_numpy = targets.data.cpu().numpy().tolist()
            label_list.extend(label_numpy)
            preds_numpy = outputs.max(1)[1].cpu().numpy().tolist()
            preds_list.extend(preds_numpy)
            out_softmax = F.softmax(outputs, dim=1)
            _, out_preds = torch.max(out_softmax, dim=1)
            testing_acc = testing_acc + torch.sum(out_preds == targets)
            label_arr = np.array(label_list)
            preds_arr = np.array(preds_list)
            cfm = confusion_matrix(label_arr, preds_arr)  #####生成混淆矩阵
    test_acc = testing_acc.double() / length
    print('The test acc is {} length {}'.format(test_acc, length))
    return cfm


# heatmap
def heatmap(cfm, mods, dir):
    plt.clf()
    # get_ipython().magic('matplotlib inline')
    f, ax1 = plt.subplots(figsize=(24, 16))
    cmap = sns.cubehelix_palette(6, start=3, rot=0, dark=0.3, light=0.90, as_cmap=True)
    sns.heatmap(cfm, linewidths=0.05, ax=ax1, cmap=cmap, annot=True, center=None, annot_kws={'size': 15})
    ax1.set_xlabel('preds', fontsize=15)
    ax1.set_ylabel('label', fontsize=15)
    ax1.set_xticklabels(mods, fontsize=10)  # 设置x轴图例为空值
    ax1.set_yticklabels(mods, fontsize=10)  # 设置x轴图例为空值
    map_dir = os.path.join(dir, 'heatmap.png')
    plt.savefig(map_dir, dpi=400)
    plt.show()


# loss和acc曲线
def loss_acc(dir, params):
    train_lloss = params['train_lloss']
    train_uloss = params['train_uloss']
    train_loss = params['train_loss']
    train_lacc = params['train_lacc']
    train_uacc = params['train_uacc']
    train_acc = params['train_acc']
    valid_lloss = params['valid_lloss']
    valid_lacc = params['valid_lacc']
    plt.clf()
    plt.plot(train_lloss, label='Train lloss')
    plt.plot(train_uloss, label='Train uloss')
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_lloss, label='valid lloss')
    plt.ylabel('Loss', fontsize=14)  # 标签
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.savefig(dir / "loss.png")
    plt.show()
    plt.clf()
    plt.plot(train_lacc, label='Train lacc')
    plt.plot(train_uacc, label='Train uacc')
    plt.plot(train_acc, label='Train acc')
    plt.plot(valid_lacc, label='valid lacc')
    plt.ylabel('Acc', fontsize=14)  # 标签
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.savefig(dir / "acc.png")
    plt.show()
