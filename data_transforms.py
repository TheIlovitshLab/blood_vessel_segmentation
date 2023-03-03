import matplotlib.pyplot as plt
import torch
import numpy as np
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


class NegativeImage(object):
    """Returns the negative of the image"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        new_image = np.max(image) - image

        # fig, axs = plt.subplots(1, 2, figsize=(18, 4))
        # axs[0].imshow(image)
        # axs[0].set_title('original')
        # axs[1].imshow(new_image)
        # axs[1].set_title('negative')
        # plt.show()

        return {'image': new_image, 'mask': mask}


class ClaheImage(object):
    """Clahe on image.

    Args: clip limit, tileGridSize
    """

    def __init__(self, clipLimit=4., tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        new_image = clahe.apply(image)

        # fig, axs = plt.subplots(1, 2, figsize=(18, 4))
        # fig.suptitle('original')
        # axs[0].imshow(image)
        # axs[1].hist(image.ravel(), np.max(image),[0,np.max(image)])
        # plt.show()
        #
        # fig, axs = plt.subplots(1, 2, figsize=(18, 4))
        # fig.suptitle('After Clahe')
        # axs[0].imshow(new_image)
        # axs[1].hist(new_image.ravel(),np.max(new_image),[0,np.max(new_image)])
        # plt.show()

        return {'image': new_image, 'mask': mask}


class RescalePixels(object):
    """Rescale the image's pixels value to a given range.

    Args:
        new_range = (new_min, new_max) (tuple): Desired pixels range.
    """

    def __init__(self, new_range):
        assert isinstance(new_range, tuple)
        self.new_min = new_range[0]
        self.new_max = new_range[1]

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        old_min = np.min(image)
        old_max = np.max(image)

        new_image = (((image - old_min) * (self.new_max - self.new_min)) / (old_max - old_min)) + self.new_min
        new_image = np.max(new_image) - new_image

        # fig, axs = plt.subplots(1, 2, figsize=(18, 4))
        # axs[0].imshow(image)
        # axs[0].set_title('original')
        # axs[1].imshow(new_image)
        # axs[1].set_title('after scalse to [0 1]')
        # plt.show()

        return {'image': new_image, 'mask': mask}


class ExpendDim(object):
    """expend dim from [batch, *image_size] to [1, batch, *image_size]"""

    def __call__(self, sample):
        return {'image': np.expand_dims(sample['image'], axis=0),
                'mask': np.expand_dims(sample['mask'], axis=0)}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image'].astype(float)).float(),
                'mask': torch.from_numpy(sample['mask'].astype(float)).float()}
