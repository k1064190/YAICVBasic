import os
import sys

import joblib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CityscapeDataset(Dataset):
    def __init__(self, img_file, label_model):
        self.img_file = img_file
        self.img_list = os.listdir(img_file)
        self.label_model = label_model

    def __len__(self):
        return len(self.img_list)

    # index를 입력받아 dataset의 index번째 원소의 (original_img, label_class)을 반환
    def __getitem__(self, index):
        img_filenum = self.img_list[index]
        img_filepath = os.path.join(self.img_file, img_filenum)
        img = Image.open(img_filepath)
        original_img, label_img = self._split_image(img)
        label_class = self.label_model.predict(label_img.reshape(-1, 3)).reshape(256, 256)
        label_class = torch.Tensor(label_class).long()
        original_img = self._transform(original_img)
        return original_img, label_class

    # 이어져 있는 원본 이미지와 라벨 이미지를 분리한다.
    def _split_image(self, img):
        img = np.array(img)
        original_img, label_img = img[:, :256, :], img[:, 256:, :]
        return original_img, label_img


    def _transform(self, img):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return transform_ops(img)