import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as function
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import kagglehub
# 下载最新版本的数据集
path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
print("Path to dataset files:", path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_root = path
train_dir = os.path.join(data_root, "casting_data", "casting_data", "train")
test_dir  = os.path.join(data_root, "casting_data", "casting_data", "test")

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # 转成3通道，适配 in_channels=3 的网络
    transforms.Resize((224, 224)),                  # 或者 (128,128)/(300,300)，看你想用多大的输入
    transforms.ToTensor(),                        # [0,255] -> [0,1]
    transforms.RandomHorizontalFlip(p=0.4),   # 可选增强：水平翻转
    transforms.RandomRotation(degrees=10),    # 可选增强：小角度旋转
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # 转成3通道，适配 in_channels=3 的网络
    transforms.Resize((224, 224)),                  # 或者 (128,128)/(300,300)，看你想用多大的输入
    transforms.ToTensor(),                        # [0,255] -> [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

training_data = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_data = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#print("Classes:", training_data.classes) '
for X, y in train_dataloader:
    print("Batch X shape:", X.shape)  # [B, 3, 224, 224]
    print("Batch y shape:", y.shape)  # [B]
    break

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
