from dataset import SegmentationDataset
from unet import UNet
import data_transforms as d_tf
import config

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms as tf
from imutils import paths
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from datetime import *
import time
import numpy as np
from argparse import ArgumentParser


class UnetSegmentationModel:

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.model = UNet().to(self.config.device)
        self.loss_fn = BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.init_lr)


    def data_loader(self, partition='train'):

        if partition == 'train':

            # load the image and mask filepaths
            image_dir = sorted(list(paths.list_images(self.config.images_dir)))
            mask_dir = sorted(list(paths.list_images(self.config.masks_dir)))

            # split data to training and testing
            split = train_test_split(image_dir, mask_dir, test_size=self.config.test_split, random_state=42)
            train_images, test_images = split[:2]
            train_masks, test_masks = split[2:]

            print('saving testing image and mask paths')
            test_images_dir = os.path.sep.join([self.model_dir, 'images_paths.txt'])
            test_masks_dir = os.path.sep.join([self.model_dir, 'masks_paths.txt'])

            f = open(test_images_dir, 'w')
            f.write('\n'.join(test_images))
            f.close()

            f = open(test_masks_dir, 'w')
            f.write('\n'.join(test_masks))
            f.close()

            # transforms = tf.Compose([tf.ToPILImage(),
            #                          tf.CenterCrop(1200),
            #                          tf.Resize((config.input_image_h, config.input_image_w)),
            #                          tf.ToTensor()])

            transforms = tf.Compose([d_tf.CenterCrop(1200),
                                     d_tf.Rescale((config.input_image_h, config.input_image_w)),
                                     d_tf.RescalePixels((0, 1)),
                                     d_tf.ExpendDim(),
                                     d_tf.ToTensor()])

            train_data = SegmentationDataset(image_dir=train_images, mask_dir=train_masks, transforms=transforms)
            test_data = SegmentationDataset(image_dir=test_images, mask_dir=test_masks, transforms=transforms)
            print('found {} examples in the training set...'.format(len(train_data)))
            print('found {} examples in the test set...'.format(len(test_data)))

            train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size,
                                      pin_memory=config.pin_memory, num_workers=os.cpu_count())
            test_loader = DataLoader(test_data, shuffle=False, batch_size=1, pin_memory=config.pin_memory,
                                     num_workers=os.cpu_count())

            train_loader.len = len(train_data)
            test_loader.len = len(test_data)

            self.train_loader, self.test_loader = train_loader, test_loader
            # return train_loader, test_loader

        elif partition == 'test':
            images = open(os.path.join(self.model_dir, self.config.test_images_dir)).read().strip().split("\n")
            masks = open(os.path.join(self.model_dir, self.config.test_masks_dir)).read().strip().split("\n")

            # transforms = tf.Compose([tf.ToPILImage(),
            #                          tf.CenterCrop(1200),
            #                          tf.Resize((config.input_image_h, config.input_image_w)),
            #                          tf.ToTensor()])

            transforms = tf.Compose([d_tf.CenterCrop(1200),
                                     d_tf.Rescale((config.input_image_h, config.input_image_w)),
                                     d_tf.RescalePixels((0, 1)),
                                     d_tf.ExpendDim(),
                                     d_tf.ToTensor()])

            test_data = SegmentationDataset(image_dir=images, mask_dir=masks, transforms=transforms)
            test_loader = DataLoader(test_data, shuffle=False, batch_size=1, pin_memory=config.pin_memory,
                                     num_workers=os.cpu_count())
            test_loader.len = len(test_data)

            self.test_loader = test_loader
            # return test_loader


    def train_model(self):
        self.model.train()

        all_loss = []
        all_mse = []
        all_acc = []
        all_jacc = []

        # for (image, mask) in train_loader:
        #     (image, mask) = (image.to(config.device), mask.to(config.device))
        for batch in self.train_loader:
            image, mask = batch['image'], batch['mask']
            image, mask = image.to(self.config.device), mask.to(self.config.device)

            pred = self.model(image)
            loss = self.loss_fn(pred, mask.float())
            mse, acc, jacc, _ = self.model_metrics(pred, mask)

            all_loss.append(loss)
            all_mse.append(mse)
            all_acc.append(acc)
            all_jacc.append(jacc)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        all_loss = sum(all_loss)/len(all_loss)
        all_mse = sum(all_mse)/len(all_mse)
        all_acc = sum(all_acc)/len(all_acc)
        all_jacc = sum(all_jacc)/len(all_jacc)

        self.update_train_metrics(all_loss, all_mse, all_acc, all_jacc)

        # return all_loss, all_mse, all_acc, all_jacc


    def eval_model(self, epoch, plot=False):
        with torch.no_grad():
            self.model.eval()

            all_loss = []
            all_mse = []
            all_acc = []
            all_jacc = []

            for batch in self.test_loader:
                image, mask = batch['image'], batch['mask']
                image, mask = image.to(config.device), mask.to(config.device)

                pred = self.model(image)
                loss = self.loss_fn(pred, mask.float())
                mse, acc, jacc, pred_th = self.model_metrics(pred, mask)

                all_loss.append(loss)
                all_mse.append(mse)
                all_acc.append(acc)
                all_jacc.append(jacc)

            all_loss = sum(all_loss)/len(all_loss)
            all_mse = sum(all_mse)/len(all_mse)
            all_acc = sum(all_acc) / len(all_acc)
            all_jacc = sum(all_jacc) / len(all_jacc)

            self.test_loss = all_loss
            self.update_test_metrics(all_loss, all_mse, all_acc, all_jacc)

            if plot:
                self.plot_preds(image, pred, pred_th, mask, epoch, partition='test')

        # return all_loss, all_mse, all_acc, all_jacc


    def init_metrics(self):
        self.train_metrics = {'train_loss': [], 'train_mse': [], 'train_acc': [], 'train_jacc': []}
        self.test_metrics = {'test_loss': [], 'test_mse': [], 'test_acc': [], 'test_jacc': []}


    def update_train_metrics(self, train_loss, train_mse, train_acc, train_jacc):
        self.train_metrics['train_loss'].append(train_loss.cpu().detach().numpy())
        self.train_metrics['train_mse'].append(train_mse.cpu().detach().numpy())
        self.train_metrics['train_acc'].append(train_acc)
        self.train_metrics['train_jacc'].append(train_jacc)
        print('train: loss {:.4f}, mse {:.4f}, acc {:.4f}, jacc {:.4f}'.format(train_loss, train_mse, train_acc, train_jacc))


    def update_test_metrics(self, test_loss, test_mse, test_acc, test_jacc):
        self.test_metrics['test_loss'].append(test_loss.cpu().detach().numpy())
        self.test_metrics['test_mse'].append(test_mse.cpu().detach().numpy())
        self.test_metrics['test_acc'].append(test_acc)
        self.test_metrics['test_jacc'].append(test_jacc)
        print('test: loss {:.4f}, mse {:.4f}, acc {:.4f}, jacc {:.4f}'.format(test_loss, test_mse, test_acc, test_jacc))


    def plot_metrics(self):
        # plots dir
        plots_dir = os.path.join(model_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        for train_metric in self.train_metrics.keys():
            metric = train_metric.split('_')[-1]
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(self.train_metrics[train_metric], label=f'train {metric}')
            plt.plot(self.test_metrics[f'test_{metric}'], label=f'test {metric}')
            plt.title(f'Training {metric} on Dataset')
            plt.xlabel('Epoch #')
            plt.ylabel(metric)
            plt.legend(loc='lower left')
            plt.savefig(os.path.join(plots_dir, f'{metric}_plot'))


    def save_model(self):
        # save models
        best_dir = os.path.join(self.model_dir, f'unet_best.pth')
        torch.save(self.model, best_dir)

        last_dir = model_dir = os.path.join(self.model_dir, f'unet_last.pth')
        torch.save(self.model, last_dir)


    @staticmethod
    def model_metrics(pred, mask):
        mse = torch.mean(torch.square(pred - mask))

        pred_th = torch.clone(pred)

        th = ((torch.max(pred) + torch.min(pred)) / 2)
        pred_th[pred_th >= th] = 1
        pred_th[pred_th < th] = 0

        pred_flat = torch.reshape(pred_th, (1, -1))
        mask_flat = torch.reshape(mask, (1, -1))

        tn, fp, fn, tp = confusion_matrix(mask_flat[0].detach().numpy(), pred_flat[0].detach().numpy()).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        jacc = tp / (tp + fn + fp)

        # ToDo: plot confusion matrix
        # mat = confusion_matrix(mask_flat[0].detach().numpy(), pred_flat[0].detach().numpy()).ravel()
        # df = pd.DataFrame(mat / np.sum(mat) * 10, index=[i for i in ('vessel', 'not vessel')], columns=[i for i in ('vessel', 'not vessel')])

        return mse, acc, jacc, pred_th


    @staticmethod
    def plot_preds(image, pred, pred_th, mask, epoch, partition='test', save_dir=None, fig_num=0):

        # create fn fp color map
        diff = (mask - pred_th)[0, 0, :, :].numpy().astype(int)
        color_map = {1: np.array([255, 0, 0]),  # red = fn
                     -1: np.array([0, 0, 255]),  # blue = fp
                     0: np.array([255, 255, 255])}  # white
        patches = [mpatches.Patch(color=color_map[1] / 255, label='fn'),
                   mpatches.Patch(color=color_map[-1] / 255, label='fp')]

        color_diff = np.ndarray(shape=(diff.shape[0], diff.shape[1], 3), dtype=int)
        for i in range(0, diff.shape[0]):
            for j in range(0, diff.shape[1]):
                color_diff[i][j] = color_map[diff[i][j]]

        fig, axs = plt.subplots(1, 5, figsize=(18, 4))
        if epoch is not None:
            fig.suptitle(f'epoch {epoch} - {partition}')

        axs[0].imshow(image[0, 0, :, :])
        axs[0].set_title('image')

        axs[1].imshow(pred[0, 0, :, :])
        axs[1].set_title('pred')

        axs[2].imshow(pred_th[0, 0, :, :])
        axs[2].set_title('pred thresh')

        axs[3].imshow(mask[0, 0, :, :])
        axs[3].set_title('GT')

        axs[4].imshow(1 - mask[0, 0, :, :], cmap='Greys')
        axs[4].imshow(color_diff, alpha=0.5)
        axs[4].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        axs[4].set_title('diff')

        plt.show()

        if save_dir is not None:
            plt.draw()
            fig.savefig(os.path.join(save_dir, f'fig_{fig_num}'))


    def Train(self):

        # create loaders
        self.data_loader()

        start_t = time.time()
        self.best_loss = 100
        self.best_epoch = 0

        self. init_metrics()

        for epoch in range(config.epochs):

            print('------------')
            print('epoch {}/{}'.format(epoch, config.epochs))

            # train
            self.train_model()

            # eval
            self.eval_model(epoch, plot=False)

            # best model
            if self.test_loss < self.best_loss:
                self.best_loss = self.test_loss
                self.best_epoch = epoch

        end_t = time.time()

        # plot metrics
        self.plot_metrics()

        # save model
        self.save_model()

        print('train finished')
        print(f'total training time: {end_t - start_t}')
        print(f'best epoch {self.best_epoch}')


    def Test(self, saved_epoch, plots_dir):

        # load model
        self.model = torch.load(os.path.join(self.model_dir, f'unet_{saved_epoch}.pth')).to(config.device)

        # create loaders
        self.data_loader(partition='teat')

        self.model.eval()
        with torch.no_grad():
            all_images, all_masks, all_preds = [], [], []

            fig = 0
            for batch in self.test_loader:
                image, mask = batch['image'], batch['mask']
                image, mask = image.to(config.device), mask.to(config.device)

                pred = self.model(image)

                pred_th = torch.clone(pred)
                th = ((torch.max(pred_th) + torch.min(pred_th)) / 2)
                pred_th[pred_th >= th] = 1
                pred_th[pred_th < th] = 0

                self.plot_preds(image, pred, pred_th, mask, epoch=None, partition=None, save_dir=plots_dir, fig_num=fig)
                fig += 1

                all_images.append(image.numpy())
                all_masks.append(mask.numpy())
                all_preds.append(pred.numpy())

            all_images = np.concatenate(all_images)
            all_masks = np.concatenate(all_masks)
            all_preds = np.concatenate(all_preds)

            print('test finished')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--saved_epoch', type=str, default='best')
    args = parser.parse_args()

    if args.train:
        # create dir to save models
        ct = "{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_dir = os.path.join(config.output_dir, ct)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # train
        unet_model = UnetSegmentationModel(config, model_dir)
        unet_model.Train()

    if args.test:
        # load model
        model_dir = os.path.join(config.output_dir, args.model)

        # create plots dir
        plots_dir = os.path.join(model_dir, 'test_plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # test
        unet_model = UnetSegmentationModel(config, model_dir)
        unet_model.Test(args.saved_epoch, plots_dir)
