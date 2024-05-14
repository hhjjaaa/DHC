import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='cps')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=1)
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

from models.vnet_dst import VNet_Decoupled
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss, Onehot_WeightedCrossEntropyLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
import torch.nn.functional as F
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
    model = VNet_Decoupled(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=3e-5,
        momentum=0.9,
        nesterov=True
    )
    return model, optimizer


def shift_log(x, offset=1e-6):
    """
    First shift, then calculate log for numerical stability.
    """
    return torch.log(torch.clamp(x + offset, max=1.))





class WorstCaseEstimationLoss(nn.Module):

    def __init__(self):
        super(WorstCaseEstimationLoss, self).__init__()

    def forward(self, y_l, y_l_adv, y_u, y_u_adv):

        prediction_l = torch.argmax(y_l.detach(), dim=1, keepdim=True).long()
        if len(prediction_l.shape) == len(y_l_adv.shape):
            assert prediction_l.shape[1] == 1
            prediction_l = prediction_l[:, 0]
        loss_l = F.cross_entropy(y_l_adv, prediction_l)

        prediction_u = torch.argmax(y_u.detach(), dim=1, keepdim=True).long()
        if len(prediction_u.shape) == len(y_u_adv.shape):
            assert prediction_u.shape[1] == 1
            prediction_u = prediction_u[:, 0]
        loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)

        return 2 * loss_l + loss_u



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

    # make data loader
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

    # make loss function
    # weight,_ = labeled_loader.dataset.weight()
    # print(weight)
    # print(sum(weight))
    loss_func = make_loss_function(args.sup_loss)
    # cps_loss_func = make_loss_function(args.cps_loss)
    if args.cps_loss == 'wce':
        cps_loss_func = Onehot_WeightedCrossEntropyLoss()
    worst_case_criterion = WorstCaseEstimationLoss()

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

            if args.mixed_precision:
                with autocast():
                    output_A, output_A_pseudo, output_A_worst = model_A(image)
                    output_B, output_B_pseudo, output_B_worst = model_B(image)
                    del image

                    # sup (ce + dice)
                    output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                    output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]
                    loss_sup = loss_func(output_A_l, label_l) + loss_func(output_B_l, label_l)

                    # cps (ce only)
                    # max_A = torch.argmax(output_A, dim=1, keepdim=True).long()
                    # max_B = torch.argmax(output_B, dim=1, keepdim=True).long()

                    confidence_A, target_A = F.softmax(output_A_pseudo, dim=1).max(dim=1)
                    confidence_B, target_B = F.softmax(output_B_pseudo, dim=1).max(dim=1)

                    # target_A = F.gumbel_softmax(output_A_pseudo, dim=1, hard=True)
                    # target_B = F.gumbel_softmax(output_B_pseudo, dim=1, hard=True)
                    # print(target_A.shape)

                    # print(confidence_A)
                    mask_A = (confidence_A > 0.95).float()
                    mask_B = (confidence_B > 0.95).float()

                    loss_cps = cps_loss_func(output_A, target_B, mask_B) + cps_loss_func(output_B, target_A, mask_A)


                    output_A_l_adv, output_A_u_adv = output_A_worst[:tmp_bs, ...], output_A_worst[tmp_bs:, ...]
                    output_B_l_adv, output_B_u_adv = output_B_worst[:tmp_bs, ...], output_B_worst[tmp_bs:, ...]


                    loss_worst_case = worst_case_criterion(output_A_l, output_A_l_adv, output_A_u, output_A_u_adv) + \
                                      worst_case_criterion(output_B_l, output_B_l_adv, output_B_u, output_B_u_adv)


                    loss = loss_sup + cps_w * (loss_cps + loss_worst_case)
                    # loss = loss_sup + cps_w * loss_cps

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()
                model_A.step()
                model_B.step()

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
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')

        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 10 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    output = (model_A(image) + model_B(image)) / 2.0
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

    writer.close()
