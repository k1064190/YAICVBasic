import os
import sys

import joblib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from UNet256 import UNet
from CityscapeDataset import CityscapeDataset

train_dir = "../datasets/cityscapes_data/train"
val_dir = "../datasets/cityscapes_data/val"
model_path = "./UNet_driving10.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_classes = 10
model = UNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path))
print("model loaded")

label_model = joblib.load("./kmeans_model.pkl")
print("label_model loaded")

test_batch_size = 8
dataset = CityscapeDataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size=test_batch_size)

X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model(X)
print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred.shape)

fig, axes = plt.subplots(test_batch_size, 3, figsize=(15, test_batch_size*5))

iou_score = []
for i in range(test_batch_size):
    landscape = X[i].permute(1, 2, 0).cpu().detach().numpy() * 0.5 + 0.5
    label_class = Y[i].cpu().detach().numpy()
    label_class_pred = Y_pred[i].cpu().detach().numpy()

    # iou_score
    intersection = np.logical_and(label_class, label_class_pred)
    union = np.logical_or(label_class, label_class_pred)
    iou_score.append(np.sum(intersection) / np.sum(union))

    axes[i, 0].imshow(landscape)
    axes[i, 1].imshow(label_class)
    axes[i, 2].imshow(label_class_pred)

plt.show()
print("iou_score: ", iou_score)
