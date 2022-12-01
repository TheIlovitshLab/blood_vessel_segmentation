import torch
import os

data_dir = os.path.join('dataset', 'train')
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')

test_split = 0.15

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pin_memory = True if device == 'cuda' else False

num_channels = 1
num_classes = 1
num_levels = 3

init_lr = 0.001
epochs = 2
batch_size = 6

input_image_w = 1200
input_image_h = 1200

threshold = 0.5

output_dir = 'output'

# model_dir = os.path.join(output_dir, 'unet_best.pth')
# plots_dir = os.path.sep.join([output_dir, 'plot.png'])
test_images_dir = 'images_paths.txt'
test_masks_dir = 'masks_paths.txt'