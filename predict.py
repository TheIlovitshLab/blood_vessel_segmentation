import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import tifffile as tiff
from train import data_loader, plot_preds


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


def make_predictions(model, loader, plots_dir):
    model.eval()
    with torch.no_grad():

        all_images, all_masks, all_preds = [], [], []

        fig = 0
        for batch in loader:
            image, mask = batch['image'], batch['mask']
            image, mask = image.to(config.device), mask.to(config.device)

            pred = model(image)

            pred_th = torch.clone(pred)
            th = ((torch.max(pred_th) + torch.min(pred_th)) / 2)
            pred_th[pred_th >= th] = 1
            pred_th[pred_th < th] = 0

            plot_preds(image, pred, pred_th, mask, epoch=None, partition=None, save_dir=plots_dir, fig_num=fig)
            fig += 1

            all_images.append(image.numpy())
            all_masks.append(mask.numpy())
            all_preds.append(pred.numpy())

        all_images = np.concatenate(all_images)
        all_masks = np.concatenate(all_masks)
        all_preds = np.concatenate(all_preds)

        return all_images, all_masks, all_preds


if __name__ == '__main__':

    # load model
    model_name = '20221223_133230'
    saved_epoch = 'best'

    model_dir = os.path.join(config.output_dir, model_name)
    unet = torch.load(os.path.join(model_dir, f'unet_{saved_epoch}.pth')).to(config.device)

    # create plots dir
    plots_dir = os.path.join(model_dir, 'test_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # load test files
    test_loader = data_loader(model_dir, partition='test')

    # predictions
    images, masks, preds = make_predictions(unet, test_loader, plots_dir)

    print(':)')
