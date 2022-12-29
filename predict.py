import config
import os
from main import UnetSegmentationModel
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='20221228_212746')
    parser.add_argument('--saved_epoch', type=str, default='best')
    args = parser.parse_args()

    model_dir = os.path.join(config.output_dir, args.model)
    unet_model = UnetSegmentationModel(config, model_dir)
    unet_model.Predict(args.saved_epoch)
