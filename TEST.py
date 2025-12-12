import os
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as function
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import kagglehub
# 下载最新版本的数据集
path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
print("Path to dataset files:", path)

# 2. 选设备（GPU / CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 3. 处理数据集
data_root = path
train_dir = os.path.join(data_root, "casting_data", "casting_data", "train")
test_dir  = os.path.join(data_root, "casting_data", "casting_data", "test")

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # 转成3通道，适配 in_channels=3 的网络
    transforms.Resize((256, 256)),                  # 或者 (128,128)/(300,300)，看你想用多大的输入
    transforms.ToTensor(),                        # [0,255] -> [0,1]
    transforms.RandomHorizontalFlip(p=0.4),   # 可选增强：水平翻转
    transforms.RandomRotation(degrees=10),    # 可选增强：小角度旋转
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # 转成3通道，适配 in_channels=3 的网络
    transforms.Resize((256, 256)),                  # 或者 (128,128)/(300,300)，看你想用多大的输入
    transforms.ToTensor(),                        # [0,255] -> [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

training_data = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_data = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)
#print("classes:", training_data.classes)  # ['def_front', 'ok_front'] 之类

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for X, y in train_dataloader:
    print("X shape:", X.shape)   # [32, 3, 256, 256]
    print("y shape:", y.shape)   # [32]
    break

class my2D_CNN(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        # 空间下采样：两次 MaxPool2d(2,2)
        self.pool = nn.MaxPool2d(2, 2)
        # 尺寸变化：
        # 256x256 --pool--> 128x128 --pool--> 64x64

        # 这里不要直接 GAP 到 1x1，而是保留一点空间格子
        # 每个格子对应一大块区域，比如 64x64 -> 8x8
        self.spp = nn.AdaptiveAvgPool2d((8, 8))   # 也可以改成 4x4 / 改成 MaxPool 看效果

        # 128 * 8 * 8 = 4096 维
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # [B, 1, 256, 256]
        x = self.pool(function.relu(self.bn1(self.conv1(x))))  # -> [B, 16, 128, 128]
        x = self.pool(function.relu(self.bn2(self.conv2(x))))  # -> [B, 32,  64,  64]
        x = function.relu(self.bn3(self.conv3(x)))             # -> [B, 64,  64,  64]
        x = function.relu(self.bn4(self.conv4(x)))             # -> [B, 128,  64,  64]

        # 保留 8x8 的粗空间网格
        x = self.spp(x)                                 # -> [B, 128, 8, 8]

        x = x.view(x.size(0), -1)                       # -> [B, 8192]

        x = self.dropout(function.relu(self.fc1(x)))
        x = self.fc2(x)                                 # -> [B, 2]
        return x
    
model = my2D_CNN().to(device)
print(model)



# 4. 训练与评估模型
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 24
min_epochs = 15

target_acc = 0.995   # 99.5%
target_loss = 0.025

best_acc = 0.0
best_loss = float("inf")
best_state = None

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    
    running_loss = 0.0
    running_total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        running_total += batch_size
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss = running_loss / running_total
    print(f"Train Loss: {train_loss:.6f}")
    return train_loss

def test(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # 注意这里按样本数加权，最后再除以 total
            batch_size = X.size(0)
            test_loss += loss.item() * batch_size
            correct += (pred.argmax(1) == y).sum().item()
            total += batch_size

    avg_loss = test_loss / total
    acc = correct / total

    print(f"Test Error: \n Accuracy: {100*acc:.1f}%, Avg loss: {avg_loss:.8f}\n")
    return avg_loss, acc

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    print("-------------------------------")

    # ======= 训练阶段 =======
    train_loss = train(train_dataloader, model, loss_fn, optimizer, device)

    # ======= 验证阶段 =======
    val_loss, val_acc = test(test_dataloader, model, loss_fn, device)
    
    if epoch+1 > 10:
        scheduler.step()
    # test() 函数保持之前那样：model.eval() + no_grad，返回 (avg_loss, acc)

    # 记录“整体最优”模型
    if (epoch + 1) >= min_epochs:
        if (val_acc > best_acc) or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"🌟 New best so far (after epoch {epoch+1}): acc={best_acc:.6f}, loss={best_loss:.6f}")

    # ======= Early Stopping：只在 min_epochs 之后才允许触发 =======
    if (epoch + 1) >= min_epochs and val_acc >= target_acc and val_loss <= target_loss:
        print(f"✅ Early stopping at epoch {epoch+1}: "
              f"acc={val_acc:.4f}, loss={val_loss:.6f}")
        torch.save(model.state_dict(), "model_earlystop_995acc_0035loss.pth")
        break

#保存训练模型

# 训练结束后（无论是否 early stop），把“整体最优”也存一份
if best_state is not None:
    model.load_state_dict(best_state)
    best_path = os.path.abspath("myCNN_best.pth")
    torch.save(model.state_dict(), best_path)
    print(f"Best overall model saved to: {best_path}, acc={best_acc:.4f}, loss={best_loss:.6f}")