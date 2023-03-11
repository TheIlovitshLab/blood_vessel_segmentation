import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import random
from scipy import ndimage


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), 'output size should be int or tuple'
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
        assert isinstance(output_size, (int, tuple)), 'output size should be int or tuple'
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
        assert isinstance(new_range, tuple), 'range should be tuple'
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


class RandomRotate(object):
    """Rotate the image and the mask by a random angle within a given range.

    Args:
        angle_range (tuple): the possible range of the angle.
    """

    def __init__(self, angle_range):
        assert isinstance(angle_range, (tuple)), 'range should be tuple'
        self.min_angle = angle_range[0]
        self.max_angle = angle_range[1]

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        angle = random.uniform(self.min_angle, self.max_angle)

        image = ndimage.rotate(image, angle, reshape=False)
        mask = ndimage.rotate(mask, angle, reshape=False)

        return {'image': image, 'mask': mask}


class RandomFlip(object):
    """Flip the image and the mask with probability 0.5."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if random.uniform(0, 1) >= 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        return {'image': image, 'mask': mask}


class RandomCrop(object):
    """Randomly crop a portion of the image and mask, when the relationship between width and height is preserved."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # random scale
        scale = random.uniform(0.5, 1)
        h = int(image.shape[0] * scale)
        w = int(image.shape[1] * scale)

        # random top-left point in image
        left = int(random.uniform(0, image.shape[0] - h -1))
        top = int(random.uniform(0, image.shape[1] - w - 1))

        # plt.figure()
        # plt.imshow(image)
        # plt.scatter(left, top, c='red')
        # plt.scatter(left+w, top+h, c='red')
        # plt.show()

        image = image[top:top+h, left:left+w]
        mask = mask[top:top+h, left:left+w]

        # plt.figure()
        # plt.imshow(image)
        # plt.show()
        return {'image': image, 'mask': mask}


class RandomBrightness(object):
    """ Random adjustment of the brightness  of the input image.

    Args:
        brightness_range (tuple): the possible range of the brightness adjustment.
    """

    def __init__(self, brightness_range):
        assert isinstance(brightness_range, (tuple)), 'range should be tuple'
        self.brightness_range = brightness_range

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        # image = np.clip(image * brightness, 0, 255)

        # plt.figure()
        # plt.imshow(image)
        # plt.title(f'after brightness {brightness}')
        # plt.show()

        return {'image': image, 'mask': mask}


class RandomContrast(object):
    """ Random adjustment of the contrast of the input image.

    Args:
        contrast_range (tuple): the possible range of the contrast adjustment.
    """

    def __init__(self, contrast_range):
        assert isinstance(contrast_range, (tuple)), 'range should be tuple'
        self.contrast_range = contrast_range

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # mean = np.mean(image)
        # image = np.clip((image - mean) * np.random.uniform(self.contrast_range[0], self.contrast_range[1]) + mean, 0, 255)

        contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        # image = np.clip((image - 511.5) * contrast + 511.5, 0, 1023)

        # plt.figure()
        # plt.imshow(image)
        # plt.title(f'after contrast {contrast}')
        # plt.show()

        return {'image': image, 'mask': mask}


class LinearStretch(object):
    """ Stretch the image pixles range to [0,255] - 8bit."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # plt.figure()
        # plt.imshow(image)
        # plt.title('original')
        # plt.show()

        old_min = np.min(image)
        old_max = np.max(image)
        new_min = 0
        new_max = 255
        image = ((image - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

        # plt.figure()
        # plt.imshow(image)
        # plt.title('after stretch')
        # plt.show()

        return {'image': image, 'mask': mask}


class GaussNoise(object):
    """Add Gaussian noise to image.
    
    Args:
        mean (float):
        std (float):"
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        image = image + np.random.normal(self.mean, self.std, size=image.shape)

        # plt.figure()
        # plt.imshow(image)
        # plt.title('after gaussian noise')
        # plt.show()

        return {'image': image, 'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image'].astype(float)).float(),
                'mask': torch.from_numpy(sample['mask'].astype(float)).float()}
