import sys
# -*-coding:utf-8-*-

from my_dataset import MyDataSet
from utils import read_split_data
import os
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from LSR import LSR
import csv
import cv2
# from model_ef import efficientnet_b0 as create_model
# from model_v3 import mobilenet_v3_large as create_model
from model_v3 import mobilenet_v3_small as create_model

root = "../data/train/complexion_data"  # 数据集所在根目录


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    Red = 0
    Yellow = 0
    black = 0
    red_yellow = 0
    white = 0

    for i in val_images_label:
        if i == 0:
            Red += 1
        elif i == 1:
            Yellow += 1
        elif i == 2:
            black += 1
        elif i == 3:
            red_yellow += 1
        elif i == 4:
            white += 1
    print('Red={},Yellow={},black={},red_yellow={},white={}'.format(Red, Yellow, black, red_yellow, white))

    save_path = "/home/wanghuijuan1/face/demo/data/train/complexion_val"

    for each in val_images_path:
        each1 = each.lstrip('/home/wanghuijuan1/face/demo/data/train/complexion/').replace('/', '')

        img = cv2.imread(each, 1)

        cv2.imwrite(save_path + os.sep + '%s' % each1, img)

if __name__ == '__main__':
    main()