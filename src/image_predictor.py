# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description :   Predict
-------------------------------------------------
"""
import numpy as np
import torch
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101

import os
import time
from PIL import Image, ImageOps

from dataloaders.composed_transformer import ComposedTransformer
from utils import common_util
from utils.decode_util import decode_segmap

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Predictor:
    def __init__(self, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = self.load_model()

    def load_model(self):
        model = deeplabv3_resnet101(pretrained=True)
        model.classifier[-1] = Conv2d(256, self.num_classes, 1)
        model.load_state_dict(torch.load(self.model_path, map_location='cuda:0'))
        model.eval()
        return model

    def predict(self, x):
        x = x.to(device)
        self.model.to(device)
        start_time = time.time()
        y_hat = self.model(x)['out'].cpu().detach().numpy()
        print('time:', time.time()-start_time)
        return y_hat

def predict_image(args, image,label):
    cp_transformer = ComposedTransformer(base_size=args.base_size, crop_size=args.crop_size)
    predictor = Predictor(args.model_path, args.num_classes)
    sample = {"image": Image.fromarray(image), "label": Image.fromarray(label)}
    sample = cp_transformer.transform_ts(sample)
    image = sample["image"]
    label = sample["label"]
    x = image.unsqueeze(0)

    y_hat = predictor.predict(x)  # y_hat size = [batch_size, class, height, width]
    y_hat_rgb = decode_segmap(y_hat.argmax(1)[0], args.num_classes)
    # y_hat_rgb = Image.fromarray(y_hat_rgb)

    # label = Image.fromarray(np.array(label).astype(np.uint8))
    return y_hat_rgb, label


def compare_pred_mask_and_ground_truth(args, pred_mask, ground_truth_mask, show=False):
    # y_hat_rgb = Image.fromarray(y_hat_rgb)

    # label = Image.fromarray(np.array(label).astype(np.uint8))
    pred_mask = ImageOps.expand(pred_mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    pred_mask = np.array(pred_mask).astype(np.uint8)
    ground_truth_mask = ImageOps.expand(ground_truth_mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    ground_truth_mask = np.array(ground_truth_mask).astype(np.uint8)
    hstack = np.hstack([pred_mask, ground_truth_mask])
    hstack = Image.fromarray(hstack)
    hstack.save(
        os.path.join(args.output, "pred_truth_compare_{}.png".format(time.strftime("%H-%M-%S", time.localtime()))))
    if show:
        hstack.show("pred_truth_compare")

def get_args():
    import argparse

    configs = common_util.load_config()

    parser = argparse.ArgumentParser('Pytorch Deeplabv3_resnet Predicting')
    parser.add_argument('--base-size', type=int, default=600,
                        help='base image size')
    parser.add_argument('--crop-size', type=tuple, default=(300, 400),
                        help='crop image size')
    parser.add_argument('--num-classes', type=int, default=configs['num_classes'],
                        help='number of classes')
    parser.add_argument('--output', type=str, default='output',
                        help='where the output images are saved')
    parser.add_argument('--model-path', type=str, default='../assets/trained_models/model_ep_8.pkl',
                        help='where the trained model is inputted ')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    return args


if __name__ == '__main__':
    args = get_args()


    # predict images
    image_path = "../assets/images/frame0149.jpg"
    ground_truth_path = "../assets/images/frame0149.jpg"

    # pred_mask, gt_image = predict_single_image(args, image_path, ground_truth_path)
    # compare_pred_mask_and_ground_truth(args, pred_mask, gt_image, True)
