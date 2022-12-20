import torch
import numpy as np
from torchvision import transforms as tf
import cv2


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_h, new_w))
        mask = cv2.resize(mask, (new_h, new_w))

        # h and w are swapped for mask because for images,
        # x and y axes are axis 1 and 0 respectively
        # mask = mask * [new_w / w, new_h / h]

        return {'image': image, 'mask': mask}


class CenterCrop(object):
    """Crop the image in a sample.

    Args:
        output_size int: Desired output size. square crop.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        width, height = image.shape[-1], image.shape[-2]

        # process crop width and height for max available dimension
        crop_width = self.output_size[0] if self.output_size[0] < image.shape[-1] else image.shape[-1]
        crop_height = self.output_size[1] if self.output_size[1] < image.shape[-2] else image.shape[-2]

        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)

        image = image[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
        mask = mask[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]

        return {'image': image, 'mask': mask}

class ExpendDim(object):
    """expend dim from [batch, *size] to [1, 6, *size]"""

    def __call__(selfself, sample):
        return {'image': np.expand_dims(sample['image'], axis=0),
                'mask': np.expand_dims(sample['mask'], axis=0)}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image']).float(),
                'mask': torch.from_numpy(sample['mask'])}
