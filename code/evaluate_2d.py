import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm

# 假设 config 中定义了 num_cls 和 base_dir
from utils import read_list, config
from utils.config import Config
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default="fully")
parser.add_argument('--folds', type=int, default=3)
parser.add_argument('--cps', type=str, default=None)
args = parser.parse_args()




config = Config(args.task)
import torch
import torch.nn.functional as F

def load_npy(filepath):
    return np.load(filepath)

if __name__ == '__main__':


    test_cls = [i for i in range(1, config.num_cls)]  # 假设类别编号从1开始，不包括背景
    # values 的每行存储 [dice, hd95] 的累计和
    values = np.zeros((len(test_cls), 2))
    ids_list = read_list('test',task=args.task)  # 假设此函数返回测试案例的 id 列表

    for data_id in tqdm(ids_list):
        # 加载 npy 文件
        pred_path = os.path.join("./logs", args.exp,"fold"+str(args.folds), "predictions_AB", f'{data_id}.npy')
        label_path = os.path.join(config.save_dir, 'npy', f'{data_id}+T1_A_label.npy')
        pred = load_npy(pred_path)
        label = load_npy(label_path)

        # 如有需要，可进行尺寸匹配（这里假设预测和标签尺寸应一致）
        if pred.shape != label.shape:
            # 使用 torch 的插值进行调整（这里示例调整标签到预测尺寸）
            dd, ww, hh = pred.shape
            label_tensor = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)
            # 采用 trilinear 插值（注意：2D情况下请用 bilinear）
            label_tensor = F.interpolate(label_tensor, size=(dd, ww, hh), mode='trilinear', align_corners=False)
            label = label_tensor.squeeze().numpy()

        for i in test_cls:
            pred_i = (pred == i)
            label_i = (label == i)
            # 只有当预测和标签中都有该类别时才计算指标
            if pred_i.sum() > 0 and label_i.sum() > 0:
                dice = metric.binary.dc(pred_i, label_i) * 100
                hd95 = metric.binary.hd95(pred_i, label_i)
                values[i - 1] += np.array([dice, hd95])
    # 计算所有案例的平均指标
    values /= len(ids_list)
    print("====== Dice ======")
    print(np.round(values[:,0], 1))
    print("====== HD95 ======")
    print(np.round(values[:,1], 1))
    print("Mean Dice:", np.mean(values[:,0]), "Mean HD95:", np.mean(values[:,1]))
