import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='mri')
parser.add_argument('--exp', type=str, default='cps')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str,  default='0')
parser.add_argument('--cps', type=str, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch

from models.new_unet import UNet
from utils import  read_list, maybe_mkdir, test_all_case_AB_2d
from utils.config import Config
config = Config(args.task)
#
if __name__ == '__main__':
    stride_dict = {
        0: (8, 8),
        1: (16, 16),
        2: (32, 32),
    }
    stride = stride_dict[args.speed]

    snapshot_path = f'./logs/{args.exp}/'
    test_save_path = f'./logs/{args.exp}/predictions_{args.cps}/'
    maybe_mkdir(test_save_path)

    if "fully" in args.exp:
        model = UNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
        ).cuda()
        model.eval()
        args.cps = None
    elif "ssp" in args.exp:
        model_A = UNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
        ).cuda()
        model_B = UNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,

        ).cuda()
        model_A.eval()
        model_B.eval()

    else:
        model_A = UNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
        ).cuda()
        model_B = UNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,

        ).cuda()
        model_A.eval()
        model_B.eval()


    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')

    with torch.no_grad():
        if args.cps == "AB":
            model_A.load_state_dict(torch.load(ckpt_path)["A"])
            model_B.load_state_dict(torch.load(ckpt_path)["B"])
            print(f'load checkpoint from {ckpt_path}')
            test_all_case_AB_2d(
                model_A, model_B,
                read_list(args.split, task=args.task),
                task=args.task,
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                test_save_path=test_save_path
            )
        else:
            print(f'This test need 3D code')

