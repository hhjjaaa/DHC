# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# def get_activation(activation_type):
#     activation_type = activation_type.lower()
#     if hasattr(nn, activation_type):
#         return getattr(nn, activation_type)()
#     else:
#         return nn.ReLU()
#
# def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
#     layers = []
#     layers.append(ConvBatchNorm(in_channels, out_channels, activation))
#
#     for _ in range(nb_Conv - 1):
#         layers.append(ConvBatchNorm(out_channels, out_channels, activation))
#     return nn.Sequential(*layers)
#
# class ConvBatchNorm(nn.Module):
#     """(convolution => [BN] => ReLU)"""
#
#     def __init__(self, in_channels, out_channels, activation='ReLU'):
#         super(ConvBatchNorm, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=3, padding=1)
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.activation = get_activation(activation)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         return self.activation(out)
#
# class DownBlock(nn.Module):
#     """Downscaling with maxpool convolution"""
#
#     def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
#         super(DownBlock, self).__init__()
#         self.maxpool = nn.MaxPool2d(2)
#         self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
#
#     def forward(self, x):
#         out = self.maxpool(x)
#         return self.nConvs(out)
#
# class UpBlock(nn.Module):
#     """Upscaling then conv"""
#
#     def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
#         super(UpBlock, self).__init__()
#
#         # self.up = nn.Upsample(scale_factor=2)
#         self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
#         self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
#
#     def forward(self, x, skip_x):
#         out = self.up(x)
#         x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
#         return self.nConvs(x)
#
# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=9):
#         '''
#         n_channels : number of channels of the input.
#                         By default 3, because we have RGB images
#         n_labels : number of channels of the ouput.
#                       By default 3 (2 labels + 1 for the background)
#         '''
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         # Question here
#         in_channels = 64
#         self.inc = ConvBatchNorm(n_channels, in_channels)
#         self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
#         self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
#         self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
#         self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
#         self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
#         self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
#         self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
#         self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
#         self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
#         if n_classes == 1:
#             self.last_activation = nn.Sigmoid()
#         else:
#             self.last_activation = None
#
#         self.saved_T1_feature_bg_s = None
#         self.saved_T1_feature_fg_s = None
#         self.saved_count = 0
#         self.gamma_F = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x,mask_T1=None, is_train=False):
#         # Question here
#         x = x.float()
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#
#         if is_train:
#             ema_decay=0.999
#             # 如果是训练阶段，根据 mask_DWI 计算当前的前景和背景特征
#             T1_feature_fg_s = self.masked_average_pooling(x5, (mask_T1 == 1).float())
#             T1_feature_bg_s = self.masked_average_pooling(x5, (mask_T1 == 0).float())
#
#             # 使用 EMA 更新前景特征
#             if self.saved_T1_feature_fg_s is None:
#                 self.saved_T1_feature_fg_s = T1_feature_fg_s.clone().detach()
#             else:
#                 self.saved_T1_feature_fg_s = (
#                         ema_decay * self.saved_T1_feature_fg_s +
#                         (1 - ema_decay) * T1_feature_fg_s.clone().detach()
#                 )
#
#             # 使用 EMA 更新背景特征
#             if self.saved_T1_feature_bg_s is None:
#                 self.saved_T1_feature_bg_s = T1_feature_bg_s.clone().detach()
#             else:
#                 self.saved_T1_feature_bg_s = (
#                         ema_decay * self.saved_T1_feature_bg_s +
#                         (1 - ema_decay) * T1_feature_bg_s.clone().detach()
#                 )
#
#             # 如有需要，计算全局前景和背景特征的平均值
#         avg_fg = self.saved_T1_feature_fg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]
#         avg_bg = self.saved_T1_feature_bg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]
#
#         FP = avg_fg.unsqueeze(-1).unsqueeze(-1)
#         BP = avg_bg.unsqueeze(-1).unsqueeze(-1)
#
#         # 计算相似度，得到 SSP_out_T1
#         SSP_out_T1 = self.similarity_func(x5, FP, BP)
#         ssp=nn.Softmax(dim=1)(SSP_out_T1)
#
#         if is_train:
#             fg_T1 = self.masked_average_pooling(x5, (mask_T1 == 1).float())
#             bg_T1 = self.masked_average_pooling(x5, (mask_T1 == 0).float())
#             self_similarity_fg_T1 = F.cosine_similarity(
#                 x5, fg_T1[..., None,  None], dim=1)
#             self_similarity_bg_T1 = F.cosine_similarity(
#                 x5, bg_T1[..., None,  None], dim=1)
#             self_out = torch.cat(
#                 (self_similarity_bg_T1[:, None, ...], self_similarity_fg_T1[:, None, ...]),
#                 dim=1) * 10.0
#
#
#         # 计算 Self-Support Prototype (SSP)
#         SSFP_1, SSBP_1, ASFP_1, ASBP_1, pre_out = self.SSP_func(x5, SSP_out_T1)
#
#         FP_1 = FP * 0.5 + SSFP_1 * 0.5
#
#         BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
#
#         SSP_self_out_T1 = self.similarity_func(x5, FP_1, BP_1)
#
#         attn = nn.Softmax(dim=1)(SSP_self_out_T1)
#
#
#
#         bg_attention, fg_attention = attn.split(1, dim=1)
#         fg_features, bg_features = x5.split(x5.shape[1] // 2, dim=1)
#         attended_fg = fg_features *fg_attention
#         attended_bg = bg_features * bg_attention
#         x5_atten = torch.cat([attended_fg, attended_bg], dim=1)
#
#
#
#
#         x = self.up4(x5+self.gamma_F*x5_atten, x4)
#         # print("self.gamma",self.gamma_F)
#         x = self.up3(x, x3)
#         x = self.up2(x, x2)
#         x = self.up1(x, x1)
#         if self.last_activation is not None:
#             logits = self.last_activation(self.outc(x))
#             # print("111")
#         else:
#             logits = self.outc(x)
#             # print("222")
#         # logits = self.outc(x) # if using BCEWithLogitsLoss
#         # print(logits.size())
#         if is_train:
#             return logits, self_out
#         else:
#             return logits
#
#
#
#
#     def SSP_func(self, feature_q, out):
#
#         bs = feature_q.shape[0]
#         pred_1 = out.softmax(1)
#         pre_out = pred_1
#
#         pred_1 = pred_1.view(bs, 2, -1)
#
#         pred_fg = pred_1[:, 1]
#
#         pred_bg = pred_1[:, 0]
#
#
#         fg_ls = []
#         bg_ls = []
#         fg_local_ls = []
#         bg_local_ls = []
#
#         fg_thres = torch.sigmoid(self.raw_fg_thres)
#         bg_thres = torch.sigmoid(self.raw_bg_thres)
#
#         for epi in range(bs):
#             # fg_thres = 0.5
#             # bg_thres = 0.3
#
#             cur_feat = feature_q[epi].view(feature_q.shape[1], -1)  # (channels, D*H*W)
#
#             f_h, f_w = feature_q[epi].shape[-2:]
#
#             if (pred_fg[epi] > fg_thres).sum() > 0:
#                 fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
#             else:
#                 fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]
#
#             if (pred_bg[epi] > bg_thres).sum() > 0:
#                 bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
#             else:
#                 bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]
#
#             fg_proto = fg_feat.mean(-1)
#             bg_proto = bg_feat.mean(-1)
#             fg_ls.append(fg_proto.unsqueeze(0))
#             bg_ls.append(bg_proto.unsqueeze(0))
#
#             # Normalize features，加入 eps 防止除 0
#             fg_feat_norm = fg_feat / (torch.norm(fg_feat, 2, 0, True) + 1e-8)
#             if torch.isnan(fg_feat_norm).any():
#                 print("NaN detected in fg_feat_norm in SSP_func for epi", epi)
#                 return f"NaN detected in fg_feat_norm in SSP_func for epi {epi}"
#             bg_feat_norm = bg_feat / (torch.norm(bg_feat, 2, 0, True) + 1e-8)
#             if torch.isnan(bg_feat_norm).any():
#                 print("NaN detected in bg_feat_norm in SSP_func for epi", epi)
#                 return f"NaN detected in bg_feat_norm in SSP_func for epi {epi}"
#             cur_feat_norm = cur_feat / (torch.norm(cur_feat, 2, 0, True) + 1e-8)
#             if torch.isnan(cur_feat_norm).any():
#                 print("NaN detected in cur_feat_norm in SSP_func for epi", epi)
#                 return f"NaN detected in cur_feat_norm in SSP_func for epi {epi}"
#
#             cur_feat_norm_t = cur_feat_norm.t()
#             fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0
#             bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0
#             fg_sim = fg_sim.softmax(-1)
#             bg_sim = bg_sim.softmax(-1)
#
#             fg_proto_local = torch.matmul(fg_sim, fg_feat.t())
#             bg_proto_local = torch.matmul(bg_sim, bg_feat.t())
#             fg_proto_local = fg_proto_local.t().view(feature_q.shape[1],  f_h, f_w).unsqueeze(0)
#             bg_proto_local = bg_proto_local.t().view(feature_q.shape[1],  f_h, f_w).unsqueeze(0)
#             fg_local_ls.append(fg_proto_local)
#             bg_local_ls.append(bg_proto_local)
#
#         new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
#
#         new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)
#         new_fg_local = torch.cat(fg_local_ls, 0)
#         new_bg_local = torch.cat(bg_local_ls, 0)
#
#         return new_fg, new_bg, new_fg_local, new_bg_local, pre_out
#
#     def masked_average_pooling(self, feature, mask):
#
#         mask = F.interpolate(mask, size=feature.shape[2:], mode='bilinear', align_corners=True)
#
#
#         masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
#         if torch.isnan(masked_feature).any():
#             print("NaN detected in masked_average_pooling")
#             return "NaN detected in masked_average_pooling"
#         return masked_feature
#
#     def similarity_func(self, feature_q, fg_proto, bg_proto):
#         similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
#         similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
#         out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
#         if torch.isnan(out).any():
#             print("NaN detected in similarity_func")
#             return "NaN detected in similarity_func"
#         return out

####################################################cps##################################################
import torch.nn as nn
import torch
import torch.nn.functional as F
from .Transformer import Transformer


transformer_basic_dims = 512
pos_basic_dims=28*28
mlp_dim = 1024
num_heads = 8
depth = 1
num_modals = 4
patch_size = 1


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=9, queue_size=1000):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        self.dropout = nn.Dropout2d(p=0.2, inplace=False)
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

        self.saved_T1_feature_bg_s = None
        self.saved_T1_feature_fg_s = None

        self.queue_size = queue_size
        self.queue_fg = []  # 队列存储前景特征
        self.queue_bg = []  # 队列存储背景特征

        self.gamma_F = nn.Parameter(torch.zeros(1))



        self.image_pos = nn.Parameter(torch.zeros(1, pos_basic_dims, transformer_basic_dims))

        self.token_conv = nn.Conv2d(in_channels*8, transformer_basic_dims, kernel_size=patch_size,stride=patch_size)

        self.transformer= Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)


    def forward(self, x,mask_T1=None, is_train=False):
        # print("x.shape",x.shape)
        # print("mask_T1.shape",mask_T1.shape)
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5org=x5

        B, C, H, W = x5.shape







        if is_train:
            tmp_bs = x5.shape[0] // 2
            x5_label=x5[:tmp_bs, ...]
            # x5_label = x5
            ema_decay=0.999
            # 如果是训练阶段，根据 mask_DWI 计算当前的前景和背景特征
            T1_feature_fg_s = self.masked_average_pooling(x5_label, (mask_T1 == 1).float())
            T1_feature_bg_s = self.masked_average_pooling(x5_label, (mask_T1 == 0).float())

            # 使用 EMA 更新前景特征
            if self.saved_T1_feature_fg_s is None:
                self.saved_T1_feature_fg_s = T1_feature_fg_s.clone().detach()
            else:
                self.saved_T1_feature_fg_s = (
                        ema_decay * self.saved_T1_feature_fg_s +
                        (1 - ema_decay) * T1_feature_fg_s.clone().detach()
                )
            # self.saved_T1_feature_fg_s = self.update_queue(self.queue_fg, T1_feature_fg_s)

            # 使用 EMA 更新背景特征
            if self.saved_T1_feature_bg_s is None:
                self.saved_T1_feature_bg_s = T1_feature_bg_s.clone().detach()
            else:
                self.saved_T1_feature_bg_s = (
                        ema_decay * self.saved_T1_feature_bg_s +
                        (1 - ema_decay) * T1_feature_bg_s.clone().detach()
                )
            # self.saved_T1_feature_bg_s = self.update_queue(self.queue_bg, T1_feature_bg_s)

            # 如有需要，计算全局前景和背景特征的平均值
        avg_fg = self.saved_T1_feature_fg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]
        avg_bg = self.saved_T1_feature_bg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]


        FP = avg_fg.unsqueeze(-1).unsqueeze(-1)
        BP = avg_bg.unsqueeze(-1).unsqueeze(-1)

        # 计算相似度，得到 SSP_out_T1
        SSP_out_T1 = self.similarity_func(x5, FP, BP)
        ssp=nn.Softmax(dim=1)(SSP_out_T1)

        if is_train:
            tmp_bs = x5.shape[0] // 2
            x5_label = x5[:tmp_bs, ...]
            # x5_label = x5
            fg_T1 = self.masked_average_pooling(x5_label, (mask_T1 == 1).float())
            bg_T1 = self.masked_average_pooling(x5_label, (mask_T1 == 0).float())
            self_similarity_fg_T1 = F.cosine_similarity(
                x5_label, fg_T1[..., None,  None], dim=1)
            self_similarity_bg_T1 = F.cosine_similarity(
                x5_label, bg_T1[..., None,  None], dim=1)
            self_out = torch.cat(
                (self_similarity_bg_T1[:, None, ...], self_similarity_fg_T1[:, None, ...]),
                dim=1) * 10.0


        # 计算 Self-Support Prototype (SSP)
        SSFP_1, SSBP_1, ASFP_1, ASBP_1, pre_out = self.SSP_func(x5, SSP_out_T1)

        FP_1 = FP * 0.5 + SSFP_1 * 0.5

        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        SSP_self_out_T1 = self.similarity_func(x5, FP_1, BP_1)

        attn = nn.Softmax(dim=1)(SSP_self_out_T1)



        bg_attention, fg_attention = attn.split(1, dim=1)
        fg_features, bg_features = x5.split(x5.shape[1] // 2, dim=1)
        attended_fg = fg_features *fg_attention
        attended_bg = bg_features * bg_attention
        x5 = torch.cat([attended_fg, attended_bg], dim=1)

        x5 = self.token_conv(x5).permute(0, 2, 3, 1).contiguous().view(x5.size(0), -1, transformer_basic_dims)
        x5 = self.transformer(x5, self.image_pos)
        x5 = x5.view(x5.size(0), H // patch_size, W // patch_size, transformer_basic_dims).permute(0, 3, 1,
                                                                                                   2).contiguous()




        x = self.up4(x5org+x5, x4)
        # x = self.up4(x5+x5_org, x4)
        # print("self.gamma",self.gamma_F)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.dropout(x)
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))

        else:
            logits = self.outc(x)

        if is_train:
            return logits, self_out
        else:
            return logits,pre_out,attn




    def SSP_func(self, feature_q, out):

        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pre_out = pred_1

        pred_1 = pred_1.view(bs, 2, -1)

        pred_fg = pred_1[:, 1]

        pred_bg = pred_1[:, 0]


        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []

        for epi in range(bs):
            fg_thres = 0.7
            bg_thres = 0.5

            cur_feat = feature_q[epi].view(feature_q.shape[1], -1)  # (channels, D*H*W)

            f_h, f_w = feature_q[epi].shape[-2:]

            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]

            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]

            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # Normalize features，加入 eps 防止除 0
            fg_feat_norm = fg_feat / (torch.norm(fg_feat, 2, 0, True) + 1e-8)
            if torch.isnan(fg_feat_norm).any():
                print("NaN detected in fg_feat_norm in SSP_func for epi", epi)
                return f"NaN detected in fg_feat_norm in SSP_func for epi {epi}"
            bg_feat_norm = bg_feat / (torch.norm(bg_feat, 2, 0, True) + 1e-8)
            if torch.isnan(bg_feat_norm).any():
                print("NaN detected in bg_feat_norm in SSP_func for epi", epi)
                return f"NaN detected in bg_feat_norm in SSP_func for epi {epi}"
            cur_feat_norm = cur_feat / (torch.norm(cur_feat, 2, 0, True) + 1e-8)
            if torch.isnan(cur_feat_norm).any():
                print("NaN detected in cur_feat_norm in SSP_func for epi", epi)
                return f"NaN detected in cur_feat_norm in SSP_func for epi {epi}"

            cur_feat_norm_t = cur_feat_norm.t()
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0
            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t())
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t())
            fg_proto_local = fg_proto_local.t().view(feature_q.shape[1],  f_h, f_w).unsqueeze(0)
            bg_proto_local = bg_proto_local.t().view(feature_q.shape[1],  f_h, f_w).unsqueeze(0)
            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_fg_local = torch.cat(fg_local_ls, 0)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local, pre_out

    # def SSP_func(self, feature_q, out):
    #
    #     bs = feature_q.shape[0]
    #     pred_1 = out.softmax(1)
    #     pre_out = pred_1
    #
    #     pred_1 = pred_1.view(bs, 2, -1)
    #     pred_fg = pred_1[:, 1]
    #     pred_bg = pred_1[:, 0]
    #
    #     fg_thres = torch.sigmoid(self.raw_fg_thres)
    #     # print("fg_thres", fg_thres)
    #
    #     bg_thres = torch.sigmoid(self.raw_bg_thres)
    #     # print("bg_thres", bg_thres)
    #
    #     scale_factor = 20.0  # 控制sigmoid的陡峭程度，可调节
    #
    #     fg_ls = []
    #     bg_ls = []
    #     fg_local_ls = []
    #     bg_local_ls = []
    #
    #     for epi in range(bs):
    #         cur_feat = feature_q[epi].view(feature_q.shape[1], -1)  # (channels, H*W)
    #         f_h, f_w = feature_q[epi].shape[-2:]
    #
    #         # 使用可微的权重进行计算
    #         fg_mask_weights = torch.sigmoid((pred_fg[epi] - fg_thres) * scale_factor)
    #         bg_mask_weights = torch.sigmoid((pred_bg[epi] - bg_thres) * scale_factor)
    #
    #         # fg_feat
    #         fg_feat_weighted = cur_feat * fg_mask_weights.unsqueeze(0)
    #         fg_feat_mean = fg_feat_weighted.sum(-1) / (fg_mask_weights.sum() + 1e-8)
    #
    #         # bg_feat
    #         bg_feat_weighted = cur_feat * bg_mask_weights.unsqueeze(0)
    #         bg_feat_mean = bg_feat_weighted.sum(-1) / (bg_mask_weights.sum() + 1e-8)
    #
    #         fg_proto = fg_feat_mean
    #         bg_proto = bg_feat_mean
    #
    #         fg_ls.append(fg_proto.unsqueeze(0))
    #         bg_ls.append(bg_proto.unsqueeze(0))
    #
    #         # Normalize features，加入 eps 防止除 0
    #         fg_feat_norm = fg_feat_weighted / (torch.norm(fg_feat_weighted, 2, 0, True) + 1e-8)
    #         bg_feat_norm = bg_feat_weighted / (torch.norm(bg_feat_weighted, 2, 0, True) + 1e-8)
    #         cur_feat_norm = cur_feat / (torch.norm(cur_feat, 2, 0, True) + 1e-8)
    #
    #         cur_feat_norm_t = cur_feat_norm.t()
    #         fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0
    #         bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0
    #         fg_sim = fg_sim.softmax(-1)
    #         bg_sim = bg_sim.softmax(-1)
    #
    #         fg_proto_local = torch.matmul(fg_sim, fg_feat_weighted.t())
    #         bg_proto_local = torch.matmul(bg_sim, bg_feat_weighted.t())
    #         fg_proto_local = fg_proto_local.t().view(feature_q.shape[1], f_h, f_w).unsqueeze(0)
    #         bg_proto_local = bg_proto_local.t().view(feature_q.shape[1], f_h, f_w).unsqueeze(0)
    #         fg_local_ls.append(fg_proto_local)
    #         bg_local_ls.append(bg_proto_local)
    #
    #     new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
    #     new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)
    #     new_fg_local = torch.cat(fg_local_ls, 0)
    #     new_bg_local = torch.cat(bg_local_ls, 0)
    #
    #     return new_fg, new_bg, new_fg_local, new_bg_local, pre_out

    def masked_average_pooling(self, feature, mask):

        mask = F.interpolate(mask, size=feature.shape[2:], mode='bilinear', align_corners=True)


        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature


    def similarity_func(self, feature_q, fg_proto, bg_proto):
        fg_proto = F.normalize(fg_proto, p=2, dim=1)
        bg_proto = F.normalize(bg_proto, p=2, dim=1)
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out


    def update_queue(self, queue, new_feature):
        # 将新特征添加进队列，并确保队列长度不超过预设值
        queue.append(new_feature.clone().detach())
        if len(queue) > self.queue_size:
            queue.pop(0)
        # 计算队列中所有特征的平均值
        return torch.stack(queue).mean(dim=0)





def test_unet():
    # 创建一个随机输入，假设 batch_size=1，3个通道，128×128 的空间尺寸
    input_tensor = torch.randn(4, 1, 448, 448)
    input_tensor2 = torch.randn(2,1,  448, 448)

    # 实例化 VNet 模型
    # 可根据需要选择 normalization 类型（例如 'none'、'batchnorm' 等）和是否使用 dropout
    model = UNet(n_channels=1, n_classes=2)

    # 将输入传入模型，获得输出
    output,_ = model(input_tensor,input_tensor2,is_train=True)

    # 输出输入和输出的张量形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_unet()

