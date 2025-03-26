import os
import sys
import logging

from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='mri2d')
parser.add_argument('--exp', type=str, default='cps_3.18')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--img_size', default=96, type=int)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_20p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=150)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.0001)
parser.add_argument('-g', '--gpu', type=str, default='1')
parser.add_argument('-w', '--cps_w', type=float, default=0.3)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.unet import UNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.transforms2d import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
# from data.StrongAug_2d import get_StrongAug, ToTensor, CenterCrop
from data.data_loaders import Synapse_AMOS
from utils.config import Config
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


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


# def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False,transforms_tr=None, transforms_val=None):
#     if is_training:
#         dst = dst_cls(
#             task=args.task,
#             split=split,
#             repeat=repeat,
#             unlabeled=unlabeled,
#             num_cls=config.num_cls,
#             transform=transforms_tr
#         )
#         return DataLoader(
#             dst,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=args.num_workers,
#             pin_memory=True,
#             worker_init_fn=seed_worker,
#             drop_last=True
#
#         )
#     else:
#         dst = dst_cls(
#             task=args.task,
#             split=split,
#             is_val=True,
#             num_cls=config.num_cls,
#             transform=transforms_val
#         )
#         return DataLoader(dst, pin_memory=True)

def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size, args.task),
                RandomFlip_LR(),
                RandomFlip_UD(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size, args.task),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    model = UNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,

    ).cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    # optimizer= optim.Adam(model.parameters(), lr=args.base_lr,weight_decay=3e-5)
    return model, optimizer




if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    # transforms_train_labeled = get_StrongAug(config.patch_size, 3, 0.7)
    # transforms_train_unlabeled = get_StrongAug(config.patch_size, 3, 0.7)
    # transforms_val = transforms.Compose([
    #     CenterCrop(config.patch_size),
    #     ToTensor()
    # ])
    #
    # # make data loader
    # unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True,transforms_tr=transforms_train_unlabeled)
    # labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset), transforms_tr=transforms_train_labeled)
    # eval_loader = make_loader(args.split_eval, is_training=False, transforms_val=transforms_val)
    #
    # logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    # logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)



    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = xavier_normal_init_weight(model_B)

    logging.info(optimizer_A)

    loss_func = make_loss_function(args.sup_loss)
    cps_loss_func = make_loss_function(args.cps_loss)



    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model_A.train()
        model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            target_size = tuple(s // 16 for s in label_l.shape[2:])  # 计算目标尺寸
            labelx5 = F.interpolate(label_l.float(), size=target_size, mode='nearest')
            labelx5 = labelx5.long()

            if args.mixed_precision:
                with autocast():
                    output_A, x5_A_l = model_A(image,label_l,is_train=True)
                    output_B, x5_B_l = model_B(image,label_l,is_train=True)
                    del image

                    # sup (ce + dice)
                    output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                    output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]
                    loss_sup = loss_func(output_A_l, label_l) + loss_func(output_B_l, label_l)+0.2*cps_loss_func(x5_A_l,labelx5)+0.2*cps_loss_func(x5_B_l,labelx5)
                    # loss_sup = loss_func(output_A_l, label_l) + loss_func(output_B_l, label_l)

                    # cps (ce only)
                    max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                    max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()
                    loss_cps = cps_loss_func(output_A, max_B) + cps_loss_func(output_B, max_A)
                    # loss prop
                    loss = loss_sup + cps_w * loss_cps


                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())


        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}, cpsw:{cps_w} lr: {get_lr(optimizer_A)}')


        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 1 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    output_A,pre_out_A,_ = model_A(image)
                    output_B, pre_out_B,_ = model_B(image)
                    output = (output_A + output_B) / 2.0
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)

                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            # '''
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': model_A.state_dict(),
                    'B': model_B.state_dict()
                }, save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')

            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

        if epoch_num % 5 == 0:  # 每5个epoch可视化一次
            model_A.eval()
            model_B.eval()# 切换到评估模式
            with torch.no_grad():
                all_batches = list(iter(eval_loader))
                random_batch = random.choice(all_batches)
                images, gt = fetch_data(random_batch)
                output_A, pre_out_A, atten_A = model_A(images)
                output_B, pre_out_B, atten_B = model_B(images)
                output = (output_A + output_B) / 2.0


                print("image.shape", images.shape)

                # 随机选择一个样本和深度切片
                batch_size = images.shape[0]  # 当前批次的大小
                random_sample_idx = random.randint(0, batch_size - 1)  # 随机选择样本索引

                # 获取图像的深度维度

                # 随机选择深度切片的索引

                image_slice = images[random_sample_idx].cpu()[0, :, :]  # 第一张图像切片

                # 获取 Ground Truth 切片
                label_slice = gt[random_sample_idx].cpu()[0, :, :]  # 第一张GT切片

                # 获取预测切片
                pred1_slice1 = output_A[random_sample_idx].cpu()[1, :, :]  # 第一张预测切片
                # print("sspout",SSP_out_T1.shape)
                SSP_out_T11 = pre_out_A[random_sample_idx].cpu()[1, :, :]
                SSP_out_T12 = pre_out_A[random_sample_idx].cpu()[0, :, :]

                fg_attention = atten_A[random_sample_idx].cpu()[0, :, :]
                bg_attention = atten_A[random_sample_idx].cpu()[1, :, :]

                # 创建一个图像网格
                fig, axs = plt.subplots(1, 7, figsize=(20, 15))

                # 输入图像
                axs[0].imshow(image_slice, cmap='gray')
                axs[0].set_title(f'Input Image')
                axs[0].axis('off')

                axs[1].imshow(label_slice, cmap='gray')
                axs[1].set_title('Input label_slice1')
                axs[1].axis('off')

                axs[2].imshow(pred1_slice1, cmap='gray')
                axs[2].set_title('pre_out')
                axs[2].axis('off')

                im = axs[3].imshow(SSP_out_T11, cmap='gray')
                axs[3].set_title('SSP_out_T1')
                axs[3].axis('on')
                plt.colorbar(im, ax=axs[3])

                im = axs[4].imshow(SSP_out_T12, cmap='gray')
                axs[4].set_title('SSP_out_T12')
                axs[4].axis('on')
                plt.colorbar(im, ax=axs[4])

                im = axs[5].imshow(fg_attention, cmap='gray')
                axs[5].set_title('fg_attention')
                axs[5].axis('on')
                plt.colorbar(im, ax=axs[5])

                im = axs[6].imshow(bg_attention, cmap='gray')
                axs[6].set_title('bg_attention')
                axs[6].axis('on')
                plt.colorbar(im, ax=axs[6])

                # 将图像添加到TensorBoard
                fig.canvas.draw()
                img_tensor = torch.from_numpy(np.array(fig.canvas.renderer.buffer_rgba())).permute(2, 0, 1)[:3, :,
                             :] / 255.0
                writer.add_image(f'Validation/Predictions_{epoch_num}_image', img_tensor, epoch_num)
                plt.close(fig)

    writer.close()
