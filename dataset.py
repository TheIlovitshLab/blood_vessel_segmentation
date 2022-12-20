from torch.utils.data import Dataset
import cv2
import tifffile as tiff


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
        mask_dir = self.mask_dir[idx]
        # load the image from disk, swap its channels from BGR to RGB, and read the associated mask from disk in grayscale mode
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = tiff.imread(image_dir)
        image = image[1, :, :]  # the green channel
        image = image / 225.     # [0, 1]

        mask = cv2.imread(mask_dir, 0)
        mask[mask == 255] = 1
        # image = np.load(image_dir)
        # mask = np.load(mask_dir)

        # # convert to tensor
        # image = torch.from_numpy(np.array(image, dtype=float))
        # mask = torch.from_numpy(np.array(mask, dtype=float))

        # image = np.ndarray(image, dtype=float)
        # mask = np.ndarray(mask, dtype=float)
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
