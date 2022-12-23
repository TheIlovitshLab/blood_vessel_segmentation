import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import tifffile as tiff
from train import data_loader


def plot_results(image, mask, pred):

    fig, axs = plt.subplots(1, 3, figsize=(10, 10))

    axs[0].imshow(image[0, 0, :, :])
    axs[0].set_title('image')

    axs[1].imshow(mask[0, 0, :, :])
    axs[1].set_title('mask')

    axs[2].imshow(pred[0, 0, :, :])
    axs[2].set_title('pred')

    axs[3].imshow(torch.square(pred[0, 0, :, :] - mask[0, 0, :, :]))
    axs[3].set_title('MSE')

    plt.show()


def make_predictions(model, loader):
    model.eval()
    with torch.no_grad():

        all_images, all_masks, all_preds = [], [], []

        for batch in test_loader:
            image, mask = batch['image'], batch['mask']
            image, mask = image.to(config.device), mask.to(config.device)

            pred = model(image)
            th = ((torch.max(pred) + torch.min(pred)) / 2) * 0.7
            pred[pred >= th] = 1
            pred[pred < th] = 0

            fig, axs = plt.subplots(1, 4)

            axs[0].imshow(image[0, 0, :, :])
            axs[0].set_title('image')

            axs[1].imshow(mask[0, 0, :, :])
            axs[1].set_title('mask')

            axs[2].imshow(pred[0, 0, :, :])
            axs[2].set_title('pred')

            axs[3].imshow(torch.square(pred[0, 0, :, :] - mask[0, 0, :, :]))
            axs[3].set_title('MSE')

            plt.show()

            all_images.append(image.numpy())
            all_masks.append(mask.numpy())
            all_preds.append(pred.numpy())

        all_images = np.concatenate(all_images)
        all_masks = np.concatenate(all_masks)
        all_preds = np.concatenate(all_preds)

        return all_images, all_masks, all_preds


if __name__ == '__main__':

    # load model
    model_name = '20221220_200519'
    saved_epoch = 'last'

    model_dir = os.path.join(config.output_dir, model_name)
    unet = torch.load(os.path.join(model_dir, f'unet_{saved_epoch}.pth')).to(config.device)

    # load test files
    test_loader = data_loader(model_dir, partition='test')

    # predictions
    images, masks, preds = make_predictions(unet, test_loader)

    print(':)')
