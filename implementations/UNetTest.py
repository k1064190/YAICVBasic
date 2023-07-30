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

from UNet256 import UNet
from CityscapeDataset import CityscapeDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_classes = 10

data_dir = "../datasets/cityscapes_data/"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# train_filename_list = os.listdir(train_dir)
# sample_img_filepath = os.path.join(train_dir, train_filename_list[0])
# sample_img = Image.open(sample_img_filepath)
# plt.imshow(sample_img)
# plt.show()

num_items = 10000
color_array = np.random.choice(range(256), size=3*num_items).reshape(-1, 3)
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)
joblib.dump(label_model, 'kmeans_model.pkl')

dataset = CityscapeDataset(train_dir, label_model)

image, label_class = dataset[0]
plt.subplot(1, 2, 1)
# 이미지는 normalize된 tensor이므로 다시 원래대로 되돌려준다.
plt.imshow(image.numpy().transpose(1, 2, 0) * 0.5 + 0.5)
plt.subplot(1, 2, 2)
plt.imshow(label_class)
plt.show()

batch_size = 4
epochs = 10
lr = 0.01

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = UNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

step_losses = []
epoch_losses = []

for epoch in tqdm(range(epochs)):
    epoch_loss = 0

    for X, Y in tqdm(data_loader, total = len(data_loader), leave = False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
    epoch_losses.append(epoch_loss / len(data_loader))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(step_losses)
axes[1].plot(epoch_losses)
plt.show()

model_name = "UNet_driving10.pth"
torch.save(model.state_dict(), model_name)
