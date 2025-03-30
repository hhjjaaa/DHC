import os
import sys
import logging
from tqdm import tqdm
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='mri')
parser.add_argument('--exp', type=str, default='fully_multSSPwithCPS_without_all_module')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-s', '--split', type=str, default='train')
parser.add_argument('--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--base_lr', type=float, default=3e-2)
parser.add_argument('-g', '--gpu', type=str, default='7')
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
parser.add_argument('-w', '--cps_w', type=float, default=7)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.multSSP_fully import UNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from utils.loss import DC_and_CE_loss, SoftDiceLoss
from data.transforms2dmult import RandomCrop, CenterCrop, ToTensor, RandomFlip_UD, RandomFlip_LR
from data.data_loaders import Synapse_AMOS
from utils.config import Config
import torch.nn as nn

config = Config(args.task)

def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w



if __name__ == '__main__':
    import random

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f'patch size: {config.patch_size}')

    # model
    model = UNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
    ).cuda()

    # model = kaiming_normal_init_weight(model)

    # dataloader
    db_train = Synapse_AMOS(task=args.task,
                            split=args.split,
                            num_cls=config.num_cls,
                            transform=transforms.Compose([
                                RandomCrop(config.patch_size, args.task),
                                RandomFlip_LR(),
                                RandomFlip_UD(),
                                ToTensor(),
                            ]))
    db_eval = Synapse_AMOS(task=args.task,
                           split=args.split_eval,
                           is_val=True,
                           num_cls=config.num_cls,
                           transform=transforms.Compose([
                               CenterCrop(config.patch_size, args.task),
                               ToTensor()
                           ]))

    train_loader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    eval_loader = DataLoader(db_eval, pin_memory=True)
    logging.info(f'{len(train_loader)} itertations per epoch')

    # optimizer, scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    # optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999),eps=1e-8,)

    # loss function
    loss_func = DC_and_CE_loss()

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    best_eval = 0.0
    best_epoch = 0
    cps_w = get_current_consistency_weight(0)

    for epoch_num in range(args.max_epoch + 1):
        loss_list = []

        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            images, labels = fetch_data(batch,is_unsqueeze=False)


            image_T1 = images[:, :1, :, :]
            image_DWI = images[:, 1:, :, :]
            label_T1 = labels[:, :1, :, :]
            label_DWI = labels[:, 1:, :, :]


            target_size = tuple(s // 16 for s in image_T1.shape[2:])  # 计算目标尺寸

            label_T1_x5 = F.interpolate(label_T1.float(), size=target_size, mode='nearest')
            label_T1_x5 = label_T1_x5.long()


            target_size = tuple(s // 16 for s in image_DWI.shape[2:])  # 计算目标尺寸
            label_DWI_x5 = F.interpolate(label_DWI.float(), size=target_size, mode='nearest')
            label_DWI_x5 = label_DWI_x5.long()


            image = torch.cat([image_T1, image_DWI], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():
                    output, x5_T1, x5_DWI= model(x=image,mask_T1= label_T1,mask_DWI=label_DWI, is_train=True)
                    del image

                    output_T1, output_DWI = output[:tmp_bs, ...], output[tmp_bs:, ...]

                    # loss_T1 = loss_func(output_T1, label_T1) + loss_func(x5_T1, label_T1_x5)*0.2
                    # loss_DWI = loss_func(output_DWI, label_DWI) + loss_func(x5_DWI, label_DWI_x5) * 0.2
                    loss_DWI = loss_func(output_DWI, label_DWI)
                    loss_T1 = loss_func(output_T1, label_T1)
                    loss = loss_T1 + loss_DWI


                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
            else:
                print("mixed_precision is OFF")
                # raise NotImplementedError

            loss_list.append(loss.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss', np.mean(loss_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')

        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)


        # if epoch_num % 1 == 0:
        #
        #     # ''' ===== evaluation
        #     dice_list = [[] for _ in range(config.num_cls - 1)]
        #     model.eval()
        #     dice_func = SoftDiceLoss(smooth=1e-8)
        #     for batch in eval_loader:
        #         images, gt = fetch_data(batch)
        #         tmp_bs = images.shape[0] // 2
        #
        #
        #         image_T1 = images[:, :1, :, :, :]
        #         image_DWI = images[:, 1:, :, :, :]
        #         label_T1 = gt[:, :1, :, :, :]
        #         label_DWI = gt[:, 1:, :, :, :]
        #
        #         image = torch.cat([image_T1, image_DWI], dim=0)
        #
        #         output= model(image)
        #
        #         output_T1, output_DWI = output[:tmp_bs, ...], output[tmp_bs:, ...]
        #
        #         shp = output_T1.shape
        #         gt = gt.long()
        #         y_onehot = torch.zeros(shp).cuda()
        #         y_onehot.scatter_(1, label_T1, 1)
        #
        #         x_onehot = torch.zeros(shp).cuda()
        #         output_T1 = torch.argmax(output_T1, dim=1, keepdim=True).long()
        #
        #         # label_save = output[0][0].cpu().numpy().astype(np.int32)
        #         # label_save = sitk.GetImageFromArray(label_save)
        #         # sitk.WriteImage(label_save, '/home/xmli/hnwang/CLD_Semi/vis_test/label.nii.gz')
        #
        #         x_onehot.scatter_(1, output_T1, 1)
        #         dice = dice_func(x_onehot, y_onehot, is_training=False)
        #         dice = dice.data.cpu().numpy()
        #         for i, d in enumerate(dice):
        #             dice_list[i].append(d)
        #
        #     dice_mean = []
        #     for dice in dice_list:
        #         dice_mean.append(np.mean(dice))
        #     logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
        #     if np.mean(dice_mean) > best_eval:
        #         best_eval = np.mean(dice_mean)
        #         best_epoch = epoch_num
        #         save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
        #         torch.save(model.state_dict(), save_path)
        #         logging.info(f'save model to {save_path}')
        #     logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
        if epoch_num % 1 == 0:

            # ===== evaluation
            dice_list_T1 = [[] for _ in range(config.num_cls - 1)]
            dice_list_DWI = [[] for _ in range(config.num_cls - 1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                images, gt = fetch_data(batch,is_unsqueeze=False)


                image_T1 = images[:, :1, :, :]
                image_DWI = images[:, 1:, :, :]
                label_T1 = gt[:, :1, :, :]
                label_DWI = gt[:, 1:, :, :]
                image=torch.cat([image_T1, image_DWI], dim=0)

                output = model(image)


                tmp_bs = image.shape[0] // 2
                output_T1, output_DWI = output[:tmp_bs, ...], output[tmp_bs:, ...]


                # T1 evaluation
                shp_T1 = output_T1.shape
                y_onehot_T1 = torch.zeros(shp_T1).cuda()

                y_onehot_T1.scatter_(1, label_T1, 1)
                x_onehot_T1 = torch.zeros(shp_T1).cuda()
                output_T1_argmax = torch.argmax(output_T1, dim=1, keepdim=True).long()
                x_onehot_T1.scatter_(1, output_T1_argmax, 1)
                dice_T1 = dice_func(x_onehot_T1, y_onehot_T1, is_training=False)
                dice_T1 = dice_T1.data.cpu().numpy()
                for i, d in enumerate(dice_T1):
                    dice_list_T1[i].append(d)

                # DWI evaluation
                shp_DWI = output_DWI.shape
                y_onehot_DWI = torch.zeros(shp_DWI).cuda()
                y_onehot_DWI.scatter_(1, label_DWI, 1)
                x_onehot_DWI = torch.zeros(shp_DWI).cuda()
                output_DWI_argmax = torch.argmax(output_DWI, dim=1, keepdim=True).long()
                x_onehot_DWI.scatter_(1, output_DWI_argmax, 1)
                dice_DWI = dice_func(x_onehot_DWI, y_onehot_DWI, is_training=False)
                dice_DWI = dice_DWI.data.cpu().numpy()
                for i, d in enumerate(dice_DWI):
                    dice_list_DWI[i].append(d)

            dice_mean_T1 = [np.mean(lst) for lst in dice_list_T1]
            dice_mean_DWI = [np.mean(lst) for lst in dice_list_DWI]
            mean_dice_T1 = np.mean(dice_mean_T1)
            mean_dice_DWI = np.mean(dice_mean_DWI)
            logging.info(f'evaluation epoch {epoch_num}, T1 dice: {mean_dice_T1}, {dice_mean_T1}')
            logging.info(f'evaluation epoch {epoch_num}, DWI dice: {mean_dice_DWI}, {dice_mean_DWI}')
            overall_mean = (mean_dice_T1 + mean_dice_DWI) / 2.0
            if overall_mean > best_eval:
                best_eval = overall_mean
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'save model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')

    writer.close()


