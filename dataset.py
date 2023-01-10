from torch.utils.data import Dataset
import cv2
import tifffile as tiff
import numpy as np
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_dir = self.image_dir[idx]
        image = tiff.imread(image_dir)
        image = image[1, :, :].astype(np.uint8)  # the green channel

        if self.mask_dir is not None:
            mask_dir = self.mask_dir[idx]
            mask = cv2.imread(mask_dir, 0)
            mask[mask == 255] = 1
        else:
            mask = np.zeros(image.shape)

        sample = {'image': image, 'mask': mask}

        # check to see if we are applying any transformations
        if self.transforms:
            # apply the transformations to both image and its mask
            # image = self.transforms(image)
            # mask = self.transforms(mask)
            sample = self.transforms(sample)

        # return a tuple of the image and its mask
        # return image, mask
        return sample
