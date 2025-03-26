import torch
import numpy as np
import torch.nn.functional as F

class CenterCrop(object):
    def __init__(self, output_size, task):
        """
        Args:
            output_size (tuple): 期望输出尺寸 (h, w)
            task (str): 任务类型，比如 'synapse'
        """
        self.output_size = output_size
        self.task = task

    def __call__(self, sample):
        image = sample['image']  # 假设 image 的 shape 为 (H, W)
        # 判断是否需要padding
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1]

        if padding_flag:
            ph = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            pw = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)

        ret_dict = {}
        (H, W) = image.shape

        # 计算crop起始点
        if self.task == 'synapse':
            # 对于synapse任务，假设宽度维度在预处理时会下采样一半
            h1 = int(round((H - self.output_size[0]) / 2.))
            w1 = int(round((W // 2 - self.output_size[1]) / 2.))
        else:
            h1 = int(round((H - self.output_size[0]) / 2.))
            w1 = int(round((W - self.output_size[1]) / 2.))

        for key in sample.keys():
            item = sample[key]
            # 对于synapse任务，先对图像或标签进行下采样（宽度减半）
            if self.task == 'synapse':
                h_item, w_item = item.shape
                item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
                if key == 'image':
                    item = F.interpolate(item, size=(h_item, w_item // 2), mode='bilinear', align_corners=False)
                else:
                    item = F.interpolate(item, size=(h_item, w_item // 2), mode='nearest')
                item = item.squeeze().numpy()
            if padding_flag:
                item = np.pad(item, [(ph, ph), (pw, pw)], mode='constant', constant_values=0)
            # 裁剪中心区域
            item = item[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]
            ret_dict[key] = item

        return ret_dict


class RandomCrop(object):
    '''
    随机裁剪图像
    Args:
        output_size (tuple): 期望输出尺寸 (h, w)
        task (str): 任务类型，比如 'synapse'
    '''
    def __init__(self, output_size, task):
        self.output_size = output_size
        self.task = task

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1]
        if padding_flag:
            ph = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            pw = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)

        ret_dict = {}
        (H, W) = image.shape
        # 随机选取裁剪区域的起始坐标
        if self.task == 'synapse':
            h1 = np.random.randint(0, H - self.output_size[0])
            w1 = np.random.randint(0, W // 2 - self.output_size[1])
        else:
            h1 = np.random.randint(0, H - self.output_size[0])
            w1 = np.random.randint(0, W - self.output_size[1])

        for key in sample.keys():
            item = sample[key]
            if self.task == 'synapse':
                h_item, w_item = item.shape
                item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
                if key == 'image':
                    item = F.interpolate(item, size=(h_item, w_item // 2), mode='bilinear', align_corners=False)
                else:
                    item = F.interpolate(item, size=(h_item, w_item // 2), mode='nearest')
                item = item.squeeze().numpy()
            if padding_flag:
                item = np.pad(item, [(ph, ph), (pw, pw)], mode='constant', constant_values=0)
            item = item[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]
            ret_dict[key] = item

        return ret_dict


class RandomFlip_LR:
    """
    随机左右翻转
    """
    def __init__(self, prob=0.8):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = np.flip(img, axis=1).copy()
        return img

    def __call__(self, sample):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict


class RandomFlip_UD:
    """
    随机上下翻转
    """
    def __init__(self, prob=0.8):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = np.flip(img, axis=0).copy()
        return img

    def __call__(self, sample):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict


class ToTensor(object):
    '''将 sample 中的 ndarray 转换为 Tensor'''
    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                # 生成 (1, H, W) 的tensor
                ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
            elif key == 'label':
                ret_dict[key] = torch.from_numpy(item).long()
            else:
                raise ValueError("Unsupported key: {}".format(key))
        return ret_dict
