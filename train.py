from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms as tf
from imutils import paths
from tqdm import tqdm
import torch
import time
import os
import matplotlib.pyplot as plt


def data_loader(config=config):
    # load the image and mask filepaths
    image_dir = sorted(list(paths.list_images(config.images_dir)))
    mask_dir = sorted(list(paths.list_images(config.masks_dir)))

    # split data to training and testing
    split = train_test_split(image_dir, mask_dir, test_size=config.test_split, random_state=42)
    train_images, test_images = split[:2]
    train_masks, test_masks = split[2:]

    print('saving testing image paths')
    f = open(config.test_dir, 'w')
    f.write('\n'.join(test_images))
    f.close()

    transforms = tf.Compose([tf.ToPILImage(),
                             # tf.Resize((config.input_image_h, config.input_image_w)),
                             tf.CenterCrop(1200),
                             tf.ToTensor()])

    train_data = SegmentationDataset(image_dir=train_images, mask_dir=train_masks, transforms=transforms)
    test_data = SegmentationDataset(image_dir=test_images, mask_dir=test_masks, transforms=transforms)
    print('found {} examples in the training set...'.format(len(train_data)))
    print('found {} examples in the test set...'.format(len(test_data)))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, pin_memory=config.pin_memory, num_workers=os.cpu_count())
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, pin_memory=config.pin_memory, num_workers=os.cpu_count())

    train_loader.len = len(train_data)
    test_loader.len = len(test_data)

    return train_loader, test_loader


def load_model(train_len, test_len, config=config):
    unet = UNet().to(config.device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(unet.parameters(), lr=config.init_lr)

    train_steps = train_len // config.batch_size
    test_steps = test_len // config.batch_size

    return unet, loss_fn, optimizer, train_steps, test_steps


def train_model(model, train_loader, loss_fn, optimizer):
    model.train()

    all_loss = []
    all_mse = []

    for (image, mask) in train_loader:
        (image, mask) = (image.to(config.device), mask.to(config.device))

        pred = model(image)
        loss = loss_fn(pred, mask)
        mse = torch.mean(torch.square(pred-mask))

        all_loss.append(loss)
        all_mse.append(mse)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    all_loss = sum(all_loss)/len(all_loss)
    all_mse = sum(all_mse)/len(all_mse)

    return all_loss, all_mse


def eval_model(model, test_loader, epoch, plot=False):
    with torch.no_grad():
        model.eval()

        all_loss = []
        all_mse = []

        for (image, mask) in test_loader:
            (image, mask) = (image.to(config.device), mask.to(config.device))

            pred = model(image)
            loss = loss_fn(pred, mask)
            mse = torch.mean(torch.square(pred-mask))

            all_loss.append(loss)
            all_mse.append(mse)

        all_loss = sum(all_loss)/len(all_loss)
        all_mse = sum(all_mse)/len(all_mse)

    if plot:
        fig, axs = plt.subplots(1, 4)
        fig.suptitle(f'epoch = {epoch}')

        axs[0].imshow(image[0, 0, :, :])
        axs[0].set_title('image')

        axs[1].imshow(mask[0, 0, :, :])
        axs[1].set_title('mask')

        axs[2].imshow(pred[0, 0, :, :])
        axs[2].set_title('pred')

        axs[3].imshow(torch.square(pred[0, 0, :, :]-mask[0, 0, :, :]))
        axs[3].set_title('MSE')

        plt.show()

    return all_loss, all_mse


if __name__ == '__main__':
    train_loader, test_loader = data_loader()
    model, loss_fn, optimizer, train_steps, test_steps = load_model(train_len=train_loader.len, test_len=test_loader.len)

    H = {'train_loss': [], 'test_loss': [], 'train_mse': [], 'test_mse': []}

    start_t = time.time()
    # total_train_loss = 0
    # total_test_loss = 0
    # total_train_mse = 0
    # total_test_mse = 0

    for epoch in range(config.epochs):

        print('epoch {}/{}'.format(epoch, config.epochs))

        # train
        train_loss, train_mse = train_model(model, train_loader, loss_fn, optimizer)
        # total_train_loss += train_loss
        # total_train_mse += train_mse

        # eval
        test_loss, test_mse = eval_model(model, test_loader, epoch, plot=True)
        # total_test_loss += test_loss
        # total_test_mse += test_mse

        # # calc train and test loss
        # avg_train_loss = total_train_loss / train_steps
        # avg_test_loss = total_test_loss / test_steps
        # avg_train_mse = total_train_mse / train_steps
        # avg_test_mse = total_test_mse / test_steps

        print('train: loss {:.4f}, mse {:.4f}'.format(train_loss, train_mse))
        print('test: loss {:.4f}, mse {:.4f}'.format(test_loss, test_mse))

        H['train_loss'].append(train_loss.cpu().detach().numpy())
        H['test_loss'].append(test_loss.cpu().detach().numpy())
        H['train_mse'].append(train_mse.cpu().detach().numpy())
        H['test_loss'].append(test_mse.cpu().detach().numpy())

    end_t = time.time()

    print('train finished')
    print('total training time: {}'.format(end_t-start_t))

    # plot the loss
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H['train_loss'], label='train_loss')
    plt.plot(H['test_loss'], label='test_loss')
    plt.title('Training Loss on Dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(config.plots_dir)

    # plot the mse
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H['train_mse'], label='train_mse')
    plt.plot(H['test_mse'], label='test_mse')
    plt.title('Training Loss on Dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('MSE')
    plt.legend(loc='lower left')
    plt.savefig(config.plots_dir)

    torch.save(model, config.model_dir)