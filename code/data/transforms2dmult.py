import torch
import numpy as np
import torch.nn.functional as F


class CenterCrop(object):
    def __init__(self, output_size, task):
        """
        Args:
            output_size (tuple): Desired output size (h, w)
            task (str): Task type, e.g., 'synapse'
        """
        self.output_size = output_size
        self.task = task

    def __call__(self, sample):
        image = sample['image']  # image shape will now be (2, H, W)
        # Check if padding is required
        padding_flag = image.shape[1] <= self.output_size[0] or image.shape[2] <= self.output_size[1]

        if padding_flag:
            ph = max((self.output_size[0] - image.shape[1]) // 2 + 3, 0)
            pw = max((self.output_size[1] - image.shape[2]) // 2 + 3, 0)

        ret_dict = {}
        (C, H, W) = image.shape  # C = 2 for modalities
        # Calculate crop starting point
        if self.task == 'synapse':
            h1 = int(round((H - self.output_size[0]) / 2.))
            w1 = int(round((W // 2 - self.output_size[1]) / 2.))
        else:
            h1 = int(round((H - self.output_size[0]) / 2.))
            w1 = int(round((W - self.output_size[1]) / 2.))

        for key in sample.keys():
            item = sample[key]
            # Downsampling for synapse task
            if self.task == 'synapse':
                h_item, w_item = item.shape
                item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
                if key == 'image':
                    item = F.interpolate(item, size=(h_item, w_item // 2), mode='bilinear', align_corners=False)
                else:
                    item = F.interpolate(item, size=(h_item, w_item // 2), mode='nearest')
                item = item.squeeze().numpy()

            if padding_flag:
                item = np.pad(item, [(0, 0), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

            # Crop the center region
            item = item[:, h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]
            ret_dict[key] = item

        return ret_dict

class RandomCrop(object):
    '''
    Randomly crop an image
    Args:
        output_size (tuple): Desired output size (h, w)
        task (str): Task type, e.g., 'synapse'
    '''
    def __init__(self, output_size, task):
        self.output_size = output_size
        self.task = task

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[1] <= self.output_size[0] or image.shape[2] <= self.output_size[1]
        if padding_flag:
            ph = max((self.output_size[0] - image.shape[1]) // 2 + 3, 0)
            pw = max((self.output_size[1] - image.shape[2]) // 2 + 3, 0)

        ret_dict = {}
        (C, H, W) = image.shape  # C = 2 for modalities
        # Randomly select crop starting coordinates
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
                item = np.pad(item, [(0, 0), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

            # Crop the image randomly
            item = item[:, h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]
            ret_dict[key] = item

        return ret_dict

class RandomFlip_LR:
    """
    Random left-right flip
    """
    def __init__(self, prob=0.8):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = np.flip(img, axis=2).copy()  # Flipping across width dimension
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
    Random up-down flip
    """
    def __init__(self, prob=0.8):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = np.flip(img, axis=1).copy()  # Flipping across height dimension
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
    '''Convert sample from ndarray to Tensor'''
    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                # Convert to (C, H, W) tensor where C=2 for modalities
                ret_dict[key] = torch.from_numpy(item).float()
            elif key == 'label':
                ret_dict[key] = torch.from_numpy(item).long()
            else:
                raise ValueError("Unsupported key: {}".format(key))
        return ret_dict
