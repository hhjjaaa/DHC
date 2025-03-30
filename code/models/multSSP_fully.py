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
        self.saved_DWI_feature_bg_s = None
        self.saved_DWI_feature_fg_s = None

        self.register_buffer("global_T1_feature_bg_s", torch.zeros(1, in_channels*8))
        self.register_buffer("global_T1_feature_fg_s", torch.zeros(1, in_channels*8))

        self.register_buffer("global_DWI_feature_bg_s", torch.zeros(1, in_channels * 8))
        self.register_buffer("global_DWI_feature_fg_s", torch.zeros(1, in_channels * 8))

        self.queue_size = queue_size
        self.queue_fg = []  # 队列存储前景特征
        self.queue_bg = []  # 队列存储背景特征

        self.gamma_F = nn.Parameter(torch.zeros(1))



        self.image_pos_T1 = nn.Parameter(torch.zeros(1, pos_basic_dims, transformer_basic_dims))
        self.image_pos_DWI = nn.Parameter(torch.zeros(1, pos_basic_dims, transformer_basic_dims))

        self.token_conv = nn.Conv2d(in_channels*8, transformer_basic_dims, kernel_size=patch_size,stride=patch_size)

        self.transformer_T1= Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.transformer_DWI = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                          mlp_dim=mlp_dim)


    def forward(self, x,mask_T1=None, mask_DWI=None, is_train=False):
        # print("x.shape",x.shape)
        # print("mask_T1.shape",mask_T1.shape)

        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5org=x5

        tmp_bs = x5.shape[0] // 2
        x5_T1 = x5[:tmp_bs, ...]
        x5_DWI = x5[tmp_bs:, ...]


        B, C, H, W = x5.shape







        if is_train:
            tmp_bs = x5.shape[0] // 2
            # T1####################################################
            x5_T1=x5[:tmp_bs, ...]
            ema_decay=0.999
            # 如果是训练阶段，根据 mask_DWI 计算当前的前景和背景特征
            T1_feature_fg_s = self.masked_average_pooling(x5_T1, (mask_T1 == 1).float())
            T1_feature_bg_s = self.masked_average_pooling(x5_T1, (mask_T1 == 0).float())
            # 使用 EMA 更新前景特征
            if self.saved_T1_feature_fg_s is None:
                self.saved_T1_feature_fg_s = T1_feature_fg_s.clone().detach()
            else:
                self.saved_T1_feature_fg_s = (
                        ema_decay * self.saved_T1_feature_fg_s +
                        (1 - ema_decay) * T1_feature_fg_s.clone().detach()
                )
            # 使用 EMA 更新背景特征
            if self.saved_T1_feature_bg_s is None:
                self.saved_T1_feature_bg_s = T1_feature_bg_s.clone().detach()
            else:
                self.saved_T1_feature_bg_s = (
                        ema_decay * self.saved_T1_feature_bg_s +
                        (1 - ema_decay) * T1_feature_bg_s.clone().detach()
                )
            # 如有需要，计算全局前景和背景特征的平均值
            avg_fg_T1 = self.saved_T1_feature_fg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]
            avg_bg_T1 = self.saved_T1_feature_bg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]
            self.global_T1_feature_bg_s.copy_(avg_bg_T1)
            self.global_T1_feature_fg_s.copy_(avg_fg_T1)

            # DWI####################################################

            x5_DWI = x5[tmp_bs:, ...]
            ema_decay = 0.999
            # 如果是训练阶段，根据 mask_DWI 计算当前的前景和背景特征
            DWI_feature_fg_s = self.masked_average_pooling(x5_DWI, (mask_DWI == 1).float())
            DWI_feature_bg_s = self.masked_average_pooling(x5_DWI, (mask_DWI == 0).float())
            # 使用 EMA 更新前景特征
            if self.saved_DWI_feature_fg_s is None:
                self.saved_DWI_feature_fg_s = DWI_feature_fg_s.clone().detach()
            else:
                self.saved_DWI_feature_fg_s = (
                        ema_decay * self.saved_DWI_feature_fg_s +
                        (1 - ema_decay) * DWI_feature_fg_s.clone().detach()
                )
            # 使用 EMA 更新背景特征
            if self.saved_DWI_feature_bg_s is None:
                self.saved_DWI_feature_bg_s = DWI_feature_bg_s.clone().detach()
            else:
                self.saved_DWI_feature_bg_s = (
                        ema_decay * self.saved_DWI_feature_bg_s +
                        (1 - ema_decay) * DWI_feature_bg_s.clone().detach()
                )
            # 如有需要，计算全局前景和背景特征的平均值
            avg_fg_DWI = self.saved_DWI_feature_fg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]
            avg_bg_DWI = self.saved_DWI_feature_bg_s.mean(dim=0, keepdim=True)  # 形状: [1, C]
            self.global_DWI_feature_bg_s.copy_(avg_bg_DWI)
            self.global_DWI_feature_fg_s.copy_(avg_fg_DWI)



        FP_T1 = self.global_T1_feature_bg_s.unsqueeze(-1).unsqueeze(-1)
        BP_T1 = self.global_T1_feature_fg_s.unsqueeze(-1).unsqueeze(-1)

        # 计算相似度，得到 SSP_out_T1
        SSP_out_T1 = self.similarity_func(x5_T1, FP_T1, BP_T1)

        FP_DWI = self.global_DWI_feature_bg_s.unsqueeze(-1).unsqueeze(-1)
        BP_DWI = self.global_DWI_feature_fg_s.unsqueeze(-1).unsqueeze(-1)

        # 计算相似度，得到 SSP_out_T1
        SSP_out_DWI = self.similarity_func(x5_DWI, FP_DWI, BP_DWI)


        if is_train:
            tmp_bs = x5.shape[0] // 2

            # T1
            x5_T1 = x5[:tmp_bs, ...]
            fg_T1 = self.masked_average_pooling(x5_T1, (mask_T1 == 1).float())
            bg_T1 = self.masked_average_pooling(x5_T1, (mask_T1 == 0).float())
            self_similarity_fg_T1 = F.cosine_similarity(
                x5_T1, fg_T1[..., None,  None], dim=1)
            self_similarity_bg_T1 = F.cosine_similarity(
                x5_T1, bg_T1[..., None,  None], dim=1)
            self_out_T1 = torch.cat(
                (self_similarity_bg_T1[:, None, ...], self_similarity_fg_T1[:, None, ...]),
                dim=1) * 10.0

            #DWI

            x5_DWI = x5[tmp_bs:, ...]
            fg_DWI = self.masked_average_pooling(x5_DWI, (mask_DWI == 1).float())
            bg_DWI = self.masked_average_pooling(x5_DWI, (mask_DWI == 0).float())
            self_similarity_fg_DWI = F.cosine_similarity(
                x5_DWI, fg_DWI[..., None, None], dim=1)
            self_similarity_bg_DWI = F.cosine_similarity(
                x5_DWI, bg_DWI[..., None, None], dim=1)
            self_out_DWI = torch.cat(
                (self_similarity_bg_DWI[:, None, ...], self_similarity_fg_DWI[:, None, ...]),
                dim=1) * 10.0


        # 计算 Self-Support Prototype (SSP)
        SSFP_1_T1, SSBP_1_T1, ASFP_1_T1, ASBP_1_T1, pre_out_T1 = self.SSP_func(x5_T1, SSP_out_T1)

        FP_1_T1 = FP_T1 * 0.5 + SSFP_1_T1 * 0.5

        BP_1_T1 = SSBP_1_T1 * 0.3 + ASBP_1_T1 * 0.7

        SSP_self_out_T1 = self.similarity_func(x5_T1, FP_1_T1, BP_1_T1)

        attn_T1 = nn.Softmax(dim=1)(SSP_self_out_T1)



        bg_attention_T1, fg_attention_T1 = attn_T1.split(1, dim=1)
        fg_features_T1, bg_features_T1 = x5_T1.split(x5_T1.shape[1] // 2, dim=1)
        attended_fg_T1 = fg_features_T1 *fg_attention_T1
        attended_bg_T1 = bg_features_T1 * bg_attention_T1
        x5_T1 = torch.cat([attended_fg_T1, attended_bg_T1], dim=1)

        x5_T1 = self.token_conv(x5_T1).permute(0, 2, 3, 1).contiguous().view(x5_T1.size(0), -1, transformer_basic_dims)
        x5_T1 = self.transformer_T1(x5_T1, self.image_pos_T1)
        x5_T1 = x5_T1.view(x5_T1.size(0), H // patch_size, W // patch_size, transformer_basic_dims).permute(0, 3, 1,  2).contiguous()



        # DWI ########################

        SSFP_1_DWI, SSBP_1_DWI, ASFP_1_DWI, ASBP_1_DWI, pre_out_DWI = self.SSP_func(x5_DWI, SSP_out_DWI)

        FP_1_DWI = FP_DWI * 0.5 + SSFP_1_DWI * 0.5
        BP_1_DWI = SSBP_1_DWI * 0.3 + ASBP_1_DWI * 0.7

        SSP_self_out_DWI = self.similarity_func(x5_DWI, FP_1_DWI, BP_1_DWI)
        attn_DWI = nn.Softmax(dim=1)(SSP_self_out_DWI)

        bg_attention_DWI, fg_attention_DWI = attn_DWI.split(1, dim=1)
        fg_features_DWI, bg_features_DWI = x5_DWI.split(x5_DWI.shape[1] // 2, dim=1)
        attended_fg_DWI = fg_features_DWI * fg_attention_DWI
        attended_bg_DWI = bg_features_DWI * bg_attention_DWI
        x5_DWI = torch.cat([attended_fg_DWI, attended_bg_DWI], dim=1)

        x5_DWI = self.token_conv(x5_DWI).permute(0, 2, 3, 1).contiguous().view(x5_DWI.size(0), -1,
                                                                               transformer_basic_dims)
        x5_DWI = self.transformer_DWI(x5_DWI, self.image_pos_DWI)
        x5_DWI = x5_DWI.view(x5_DWI.size(0), H // patch_size, W // patch_size, transformer_basic_dims).permute(0, 3, 1, 2).contiguous()


        x5 = torch.cat([x5_T1, x5_DWI], dim=0)





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
            return logits, self_out_T1, self_out_DWI
        else:
            return logits




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
            # fg_thres = 0.5
            # bg_thres = 0.3
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
    input_tensor3 = torch.randn(2, 1, 448, 448)

    # 实例化 VNet 模型
    # 可根据需要选择 normalization 类型（例如 'none'、'batchnorm' 等）和是否使用 dropout
    model = UNet(n_channels=1, n_classes=2)

    # 将输入传入模型，获得输出
    logits, self_out_T1, self_out_DWI = model(input_tensor,input_tensor2,input_tensor3,is_train=True)

    # 输出输入和输出的张量形状
    print("Input shape:", input_tensor.shape)
    print("logits shape:", logits.shape)
    print("self_out_T1 shape:", self_out_T1.shape)
    print("self_out_DWI shape:", self_out_DWI.shape)


if __name__ == "__main__":
    test_unet()