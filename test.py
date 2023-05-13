import torch
import os
from utils.test_tool import heatmap, loss_acc

def tester(dir, result_model, mods, data_loader, device, version=0):
    model_dir = os.path.join(dir, 'model.pth')
    params_dir = os.path.join(dir, 'params.pkl')
    model = torch.load(model_dir)
    cfm=result_model(model,data_loader,device,version)
    precision(cfm)
    #画混淆矩阵a
    cfm=cfm/torch.sum(torch.Tensor(cfm),dim=1)
    acc = torch.diagonal(cfm)
    avg_acc = torch.sum(acc,dim=0)
    print('The model test avg_acc : {}'.format(avg_acc/len(acc)))
    heatmap(cfm, mods, dir)
    # loss和acc曲线
    # params = torch.load(params_dir)
    # loss_acc(dir, params)


def doubel_test(dir, result_model, mods, data_loader, device, version=0):
    model_dir = os.path.join(dir, 'model.pth')
    ema_model_dir = os.path.join(dir, 'ema_model.pth')
    params_dir = os.path.join(dir, 'params.pkl')

    model = torch.load(model_dir)
    ema_model = torch.load(ema_model_dir)

    cfm=result_model(model,data_loader,device,version)
    #画混淆矩阵a
    cfm=cfm/torch.sum(torch.Tensor(cfm),dim=1)
    acc = torch.diagonal(cfm)
    avg_acc = torch.sum(acc,dim=0)
    print('The model test avg_acc : {}'.format(avg_acc/len(acc)))
    heatmap(cfm, mods, dir)

    ema_cfm = result_model(ema_model, data_loader, device, version)
    # 画混淆矩阵a
    ema_cfm = ema_cfm / torch.sum(torch.Tensor(ema_cfm), dim=1)
    ema_acc = torch.diagonal(ema_cfm)
    ema_avg_acc = torch.sum(ema_acc, dim=0)
    print('The ema model test avg_acc : {}'.format(ema_avg_acc / len(ema_acc)))
    # heatmap(ema_cfm, mods, dir)
    # loss和acc曲线
    # params = torch.load(params_dir)
    # loss_acc(dir, params)

def precision(confusion):
    confusion = torch.tensor(confusion).T
    correct = confusion * torch.eye(confusion.shape[0])  # torch.eye：生成对角线全1，其余部分全0的二维数组
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / (correct + incorrect)
    total_correct = correct.sum().item()
    total_incorrect = incorrect.sum().item()
    percent_correct = total_correct / (total_correct + total_incorrect)
    print('{} {}'.format(precision, percent_correct))