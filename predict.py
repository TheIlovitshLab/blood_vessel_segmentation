import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import tifffile as tiff
from main import data_loader, plot_preds


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
