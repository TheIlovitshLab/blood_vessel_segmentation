from dataset import SegmentationDataset
# from unet import UNet
from unet2 import UNet
# from unet_model import UNet
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
from sklearn import metrics
import cv2
import scipy.io as sio


class UnetSegmentationModel:

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.model = UNet().to(self.config.device)
        self.loss_fn = BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.init_lr)


    def data_loader(self, partition='train'):

        train_transforms = tf.Compose([
            # d_tf.CenterCrop(1200),
            # d_tf.SigmoidStretch(0.5),
            # d_tf.LinearStretch(),
            # d_tf.RandomBrightness((0.5, 1.5)),
            # d_tf.GaussNoise(0, 1),
            # d_tf.RandomContrast((0.5, 1.5)),
            # d_tf.RescalePixels((0, 1)),
            d_tf.RandomCrop(),
            d_tf.RandomFlip(),
            d_tf.RandomRotate((-30, 30)),
            d_tf.Rescale((512, 512)),
            # d_tf.NegativeImage(),
            # d_tf.ClaheImage(),
            # d_tf.RescalePixels((0, 1)),
            d_tf.ExpendDim(),
            d_tf.ToTensor()
        ])

        test_transforms = tf.Compose([
            # d_tf.NegativeImage(),
            # d_tf.LinearStretch(),
            # d_tf.RescalePixels((0, 1)),
            d_tf.Rescale((512, 512)),
            # d_tf.ClaheImage(),
            # d_tf.RescalePixels((0, 1)),
            d_tf.ExpendDim(),
            d_tf.ToTensor()
        ])

        if partition == 'train':

            # load the image and mask filepaths
            image_dir = sorted([os.path.join(self.config.images_dir, img) for img in os.listdir(self.config.images_dir)])
            mask_dir = sorted([os.path.join(self.config.masks_dir, mask) for mask in os.listdir(self.config.masks_dir)])

            # split data to training and testing
            split = train_test_split(image_dir, mask_dir, test_size=self.config.test_split, random_state=42)    # todo: without random_state?
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

            train_data = SegmentationDataset(image_dir=train_images, mask_dir=train_masks, transforms=train_transforms)
            test_data = SegmentationDataset(image_dir=test_images, mask_dir=test_masks, transforms=test_transforms)
            print('found {} examples in the training set...'.format(len(train_data)))
            print('found {} examples in the test set...'.format(len(test_data)))

            train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size,
                                      pin_memory=config.pin_memory, num_workers=os.cpu_count())
            test_loader = DataLoader(test_data, shuffle=False, batch_size=1,
                                     pin_memory=config.pin_memory, num_workers=os.cpu_count())

            train_loader.len = len(train_data)
            test_loader.len = len(test_data)

            self.train_loader, self.test_loader = train_loader, test_loader
            # return train_loader, test_loader

        elif partition == 'test':
            images = open(os.path.join(self.model_dir, self.config.test_images_dir)).read().strip().split("\n")
            masks = open(os.path.join(self.model_dir, self.config.test_masks_dir)).read().strip().split("\n")

            test_data = SegmentationDataset(image_dir=images, mask_dir=masks, transforms=test_transforms)
            test_loader = DataLoader(test_data, shuffle=False, batch_size=1, pin_memory=config.pin_memory,
                                     num_workers=os.cpu_count())
            test_loader.len = len(test_data)

            self.test_loader = test_loader

        elif partition == 'predict':
            # images = sorted(list(paths.list_images(self.config.pred_images_dir)))
            images = sorted([os.path.join(self.config.pred_images_dir, img) for img in os.listdir(self.config.pred_images_dir)])

            pred_data = SegmentationDataset(image_dir=images, mask_dir=None, transforms=test_transforms)
            pred_loader = DataLoader(pred_data, shuffle=False, batch_size=1, pin_memory=config.pin_memory,
                                     num_workers=os.cpu_count())
            pred_loader.len = len(pred_data)
            self.pred_loader = pred_loader


    def train_model(self):
        self.model.train()

        all_loss, all_auc = [], []
        # all_acc, all_precision, all_recall, all_dice  = [], [], [], []

        for batch in self.train_loader:
            image, mask = batch['image'], batch['mask']
            image, mask = image.to(self.config.device), mask.to(self.config.device)

            pred = self.model(image)
            loss = self.loss_fn(pred, mask)
            # loss += dice_loss(F.sigmoid(pred), mask.float(), multiclass=False)

            auc = self.model_metrics(pred, mask)

            all_loss.append(loss)
            all_auc.append(auc)
            # all_acc.append(acc)
            # all_precision.append(precision)
            # all_recall.append(recall)
            # all_dice.append(dice)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        all_loss = sum(all_loss)/len(all_loss)
        all_auc = sum(all_auc) / len(all_auc)
        # all_acc = sum(all_acc) / len(all_acc)
        # all_precision = sum(all_precision)/len(all_precision)
        # all_recall = sum(all_recall)/len(all_recall)
        # all_dice = sum(all_dice)/len(all_dice)

        self.update_train_metrics(all_loss, all_auc)


    def eval_model(self, epoch, plot=False):
        with torch.no_grad():
            self.model.eval()

            all_loss, all_auc = [], []
            # all_acc, all_precision, all_recall, all_dice = [], [], [], []

            for batch in self.test_loader:
                image, mask = batch['image'], batch['mask']
                image, mask = image.to(config.device), mask.to(config.device)

                pred = self.model(image)
                loss = self.loss_fn(pred, mask)
                auc = self.model_metrics(pred, mask)

                all_loss.append(loss)
                all_auc.append(auc)
                # all_acc.append(acc)
                # all_precision.append(precision)
                # all_recall.append(recall)
                # all_dice.append(dice)

            all_loss = sum(all_loss)/len(all_loss)
            all_auc = sum(all_auc) / len(all_auc)
            # all_acc = sum(all_acc) / len(all_acc)
            # all_precision = sum(all_precision)/len(all_precision)
            # all_recall = sum(all_recall) / len(all_recall)
            # all_dice = sum(all_dice) / len(all_dice)

            self.test_loss = all_loss
            self.test_auc = all_auc
            self.update_test_metrics(all_loss, all_auc)
            #
            # if plot:
            #     self.plot_preds(image, pred, mask, all_auc, epoch, partition='test')


    def init_metrics(self):
        self.train_metrics = {'train_loss': [], 'train_auc': []}
        self.test_metrics = {'test_loss': [], 'test_auc': []}


    def update_train_metrics(self, train_loss, train_auc):
        self.train_metrics['train_loss'].append(train_loss.cpu().detach().numpy())
        self.train_metrics['train_auc'].append(train_auc)
        # self.train_metrics['train_acc'].append(train_acc)
        # self.train_metrics['train_precision'].append(train_precision)
        # self.train_metrics['train_recall'].append(train_recall)
        # self.train_metrics['train_dice'].append(train_dice)
        # print('train: loss {:.4f}, acc {:.4f}, precision {:.4f}, recall {:.4f}, dice {:.4f}'.format(train_loss, train_acc, train_precision, train_recall, train_dice))
        print('train: loss {:.4f}, auc {:.4f}'.format(train_loss, train_auc))


    def update_test_metrics(self, test_loss, test_auc):
        self.test_metrics['test_loss'].append(test_loss.cpu().detach().numpy())
        # self.test_metrics['test_acc'].append(test_acc)
        # self.test_metrics['test_precision'].append(test_precision)
        # self.test_metrics['test_recall'].append(test_recall)
        # self.test_metrics['test_dice'].append(test_dice)
        # print('train: loss {:.4f}, acc {:.4f}, precision {:.4f}, recall {:.4f}, dice {:.4f}'.format(test_loss, test_acc, test_precision, test_recall, test_dice))
        print('test: loss {:.4f}, auc {:.4f}'.format(test_loss, test_auc))


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


    def model_metrics(self, pred, mask):

        pred = torch.sigmoid(pred)
        # pred_th[pred_th >= config.threshold] = 1
        # pred_th[pred_th < config.threshold] = 0
        # pred_th = torch.clone(pred)
        # th = ((torch.max(pred) + torch.min(pred)) / 2)
        # pred_th[pred_th >= th] = 1
        # pred_th[pred_th < th] = 0

        pred_flat = torch.reshape(pred, (1, -1))
        mask_flat = torch.reshape(mask, (1, -1))

        auc = metrics.roc_auc_score(mask_flat.detach().numpy().squeeze(), pred_flat.detach().numpy().squeeze())
        # tn, fp, fn, tp = confusion_matrix(mask_flat[0].detach().numpy(), pred_flat[0].detach().numpy()).ravel()
        # acc = (tn + tp) / (tn + fp + fn + tp)
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # dice = 2 * ((precision * recall) / (precision + recall))

        # ToDo: plot confusion matrix
        # mat = confusion_matrix(mask_flat[0].detach().numpy(), pred_flat[0].detach().numpy()).ravel()
        # df = pd.DataFrame(mat / np.sum(mat) * 10, index=[i for i in ('vessel', 'not vessel')], columns=[i for i in ('vessel', 'not vessel')])

        return auc


    def save_model(self):
        # save models and config
        best_dir = os.path.join(self.model_dir, f'unet_best.pth')
        torch.save(self.model, best_dir)

        last_dir = model_dir = os.path.join(self.model_dir, f'unet_last.pth')
        torch.save(self.model, last_dir)


    @staticmethod
    def plot_preds(image, pred, pred_th, mask, epoch, partition='test', save_dir=None, fig_name=None):
        # create fn fp color map
        diff = (mask - pred_th).astype(int)
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
            fig.suptitle(f'epoch {epoch+1} - {partition}')
        if fig_name is not None:
            fig.suptitle(f'{fig_name}')

        axs[0].imshow(image)
        axs[0].set_title('image')

        axs[1].imshow(pred)
        axs[1].set_title('pred')

        axs[2].imshow(pred_th)
        axs[2].set_title('pred thresh')

        axs[3].imshow(mask)
        axs[3].set_title('GT')

        axs[4].imshow(1 - mask, cmap='Greys')
        axs[4].imshow(color_diff, alpha=0.5)
        axs[4].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        axs[4].set_title('diff')

        plt.show()

        if save_dir is not None:
            plt.draw()
            fig.savefig(os.path.join(save_dir, fig_name))


    def Train(self):

        # create loaders
        self.data_loader()

        start_t = time.time()
        self.best_loss = 100
        self.best_epoch = 0
        self.best_auc = 0

        self. init_metrics()

        for epoch in range(config.epochs):

            print('------------')
            print('epoch {}/{}'.format(epoch+1, config.epochs))

            # train
            self.train_model()

            # eval
            self.eval_model(epoch, plot=True)

            # best model
            if self.test_auc > self.best_auc:
                self.best_auc = self.test_auc
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
        self.data_loader(partition='test')

        files = self.test_loader.dataset.image_dir
        files_names = [file.split('\\')[-1].split('.')[0] for file in files]

        self.model.eval()
        with torch.no_grad():
            all_images, all_masks, all_preds = [], [], []

            for batch in self.test_loader:
                image, mask = batch['image'], batch['mask']
                image, mask = image.to(config.device), mask.to(config.device)

                pred = self.model(image)
                pred = torch.sigmoid(pred)

                image = image.numpy()
                mask = mask.numpy()
                pred = pred.numpy()

                all_images.append(image)
                all_masks.append(mask)
                all_preds.append(pred)

            all_images = np.concatenate(all_images).squeeze()
            all_masks = np.concatenate(all_masks).squeeze()
            all_preds = np.concatenate(all_preds).squeeze()

            # find threshold
            thresholds = np.linspace(0, 1, 200)

            precision, recall = [], []
            tpr, fpr = [], []
            f_score = []
            for th in thresholds:
                all_preds_th = np.zeros(all_preds.shape)
                all_preds_th[all_preds >= th] = 1

                mask_flat = np.reshape(all_masks, (-1)).squeeze().astype(float)
                preds_flat = np.reshape(all_preds_th, (-1)).squeeze().astype(float)

                tp = (all_masks * all_preds_th).sum()
                fp = ((1 - all_masks) * all_preds_th).sum()
                tn = ((1 - all_masks) * (1 - all_preds_th)).sum()
                fn = (all_masks * (1 - all_preds_th)).sum()

                # tn, fp, fn, tp = confusion_matrix(maskd_flat, preds_flat).ravel()
                precision.append(tp/(tp+fp))
                recall.append(tp/(tp+fn))

                tpr.append(tp/(tp+fn))
                fpr.append(fp/(fp+tn))

                f_score.append((2*tp) / (2*tp + fp + fn))

            precision = np.array(precision)
            recall = np.array(recall)
            tpr = np.array(tpr)
            fpr = np.array(fpr)

            auc = metrics.auc(recall, precision)
            roc_auc = metrics.auc(fpr, tpr)

            # plot pr curve
            plt.figure()
            plt.plot(recall, precision)
            plt.legend([f'auc = {float(auc)}'])
            plt.title('PR Curve')
            plt.draw()
            plt.savefig(os.path.join(plots_dir, 'pr-curve'))

            plt.figure()
            plt.plot(fpr, tpr)
            plt.legend([f'auc = {float(roc_auc)}'])
            plt.title('ROC Curve')
            plt.draw()
            plt.savefig(os.path.join(plots_dir, 'roc-curve'))


            #chosen_th =

            for idx in range(all_preds.shape[0]):
                mask = all_masks[idx, :, :].squeeze()
                image = all_images[idx, :, :].squeeze()

                pred = all_preds[idx, :, :].squeeze()
                pred_th = np.zeros(pred.shape)
                pred_th[pred >= 0.1] = 1

                # rescale back to original image shape
                image = cv2.resize(image, (self.config.input_image_w, self.config.input_image_h), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self.config.input_image_w, self.config.input_image_h), interpolation=cv2.INTER_AREA)
                pred = cv2.resize(pred, (self.config.input_image_w, self.config.input_image_h), interpolation=cv2.INTER_AREA)
                pred_th = cv2.resize(pred_th, (self.config.input_image_w, self.config.input_image_h), interpolation=cv2.INTER_AREA)

                self.plot_preds(image, pred, pred_th, mask, epoch=None, partition=None, save_dir=plots_dir, fig_name=files_names[idx])

            print('test finished')

    def Predict(self, saved_epoch):

        # load model
        self.model = torch.load(os.path.join(self.model_dir, f'unet_{saved_epoch}.pth')).to(config.device)

        # create loaders
        self.data_loader(partition='predict')

        files = self.pred_loader.dataset.image_dir
        files_names = [file.split('\\')[-1].split('.')[0] for file in files]
        save_pred = [os.path.join(self.config.pred_images_dir, file_name+'.txt') for file_name in files_names]

        self.model.eval()
        with torch.no_grad():

            i = 0
            for batch in self.pred_loader:
                image, mask = batch['image'], batch['mask']
                image, mask = image.to(config.device), mask.to(config.device)

                pred = self.model(image)
                pred_th = torch.sigmoid(pred)
                pred_th[pred_th >= config.threshold] = 1
                pred_th[pred_th < config.threshold] = 0
                # pred_th = torch.clone(pred)
                # th = ((torch.max(pred_th) + torch.min(pred_th)) / 2)
                # pred_th[pred_th >= th] = 1
                # pred_th[pred_th < th] = 0

                # save prediction
                pred_th = pred_th[0, 0, :, :].numpy().astype(int)
                sio.savemat(save_pred[i], pred_th)
                # np.savetxt(save_pred[i], pred_th)
                # i += 1

            print('segmentation finished')


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
