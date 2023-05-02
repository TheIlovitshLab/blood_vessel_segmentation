import config
import os
from main import UnetSegmentationModel
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='20230312_211623')
    parser.add_argument('--saved_epoch', type=str, default='best')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()

    # update config
    config.pred_images_dir = args.data_dir
    config.threshold = args.threshold

    model_dir = os.path.join(config.output_dir, args.model)
    unet_model = UnetSegmentationModel(config, model_dir)
    unet_model.Predict(args.saved_epoch)
