import numpy as np
import os
import torch
import json
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Параметры карты глубины
width = 640
height = 480
dtype = np.uint16

# Диапазон значений сегментации
min_value = 700
max_value = 1285

def create_mask_from_annotations(height, width, regions):
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if region['shape_attributes']['name'] == 'polygon':
            points = np.array([
                (x, y) for x, y in zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y'])
            ], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
    return mask.astype(np.float32)

def create_mask_and_filtered_depth(depth_map, min_value, max_value):
    mask = (depth_map >= min_value) & (depth_map <= max_value)
    depth_map_filtered = np.where(mask, depth_map, 0)
    return mask.astype(np.float32), depth_map_filtered.astype(np.float32)

def load_depth_map(raw_file_path):
    return np.fromfile(raw_file_path, dtype=dtype).reshape((height, width)).astype(np.float32)

class DepthDataset(Dataset):
    def __init__(self, json_file, depth_dir, transform=None):
        with open(json_file) as f:
            self.data = json.load(f)
        self.depth_dir = depth_dir
        self.transform = transform
        self.files = list(self.data.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]
        item = self.data[file_id]
        filename = item['filename']
        depth_path = os.path.join(self.depth_dir, filename)

        depth_map = load_depth_map(depth_path)
        mask = create_mask_from_annotations(height, width, item['regions'])
        _, depth_map_filtered = create_mask_and_filtered_depth(depth_map, min_value, max_value)

        if self.transform:
            depth_map_filtered = self.transform(depth_map_filtered)
            mask = self.transform(mask)

        return depth_map_filtered, mask

# Преобразования
transform = transforms.Compose([
    transforms.ToTensor()
])

# Создание DataLoader
train_dataset = DepthDataset("train_annotations.json", "depth_maps/train", transform=transform)
val_dataset = DepthDataset("valid_annotations.json", "depth_maps/valid", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.pool0 =  nn.MaxPool2d(kernel_size=2, return_indices=True) # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.pool1 =  nn.MaxPool2d(kernel_size=2, return_indices=True) # 128 -> 64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True) # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True) # 32 -> 16
        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(kernel_size=2)# 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.upsample1 = nn.MaxUnpool2d(kernel_size=2)# 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.upsample2 = nn.MaxUnpool2d(kernel_size=2)# 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.upsample3 = nn.MaxUnpool2d(kernel_size=2)# 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            )
    def forward(self, x):
        # encoder
        enc0 = self.enc_conv0(x)
        pool0, indices_e0 = self.pool0(enc0)
        enc1 = self.enc_conv1(pool0)
        pool1, indices_e1 = self.pool1(enc1)
        enc2 = self.enc_conv2(pool1)
        pool2, indices_e2 = self.pool2(enc2)
        enc3 = self.enc_conv3(pool2)
        pool3, indices_e3 = self.pool3(enc3)
        # bottleneck
        b = self.bottleneck_conv(pool3)
        # decoder
        d0 = self.dec_conv0(torch.cat((self.upsample0(b, indices_e3), enc3), dim=1))
        d1 = self.dec_conv1(torch.cat((self.upsample1(d0, indices_e2), enc2), dim=1))
        d2 = self.dec_conv2(torch.cat((self.upsample2(d1, indices_e1), enc1), dim=1))
        d3 = self.dec_conv3(torch.cat((self.upsample3(d2, indices_e0), enc0), dim=1))
        # no activation
        return d3

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1).byte()
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    return thresholded

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history['epoch'], history['train_loss'], label='train_loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='val_loss')
    axes[1].plot(history['epoch'], history['val_score'], label='val_score')
    for i in [0, 1]:
        axes[i].legend()
    plt.show()

def save_masks(epoch, model, data_loader, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.eval()
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs > 0.5  # Binarize the output
            outputs = outputs.cpu().numpy()
            masks = masks.cpu().numpy()
            for j in range(inputs.size(0)):
                input_image = inputs[j].cpu().numpy().squeeze()
                output_mask = outputs[j].squeeze()
                true_mask = masks[j].squeeze()
                plt.imsave(f"{folder}/epoch_{epoch}_batch_{i}_input_{j}.png", input_image, cmap='gray')
                plt.imsave(f"{folder}/epoch_{epoch}_batch_{i}_output_{j}.png", output_mask, cmap='gray')
                plt.imsave(f"{folder}/epoch_{epoch}_batch_{i}_true_{j}.png", true_mask, cmap='gray')

def train(model, opt, loss_fn, epochs, data_tr, data_val):
    history = {'epoch':[],'train_loss':[],'val_loss':[],'val_score':[]}
    X_val, Y_val = next(iter(data_val))
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            # data to device
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # set parameter gradients to zero
            opt.zero_grad()
            # forward
            Y_pred = model(X_batch)
            loss =  loss_fn(Y_batch, Y_pred) # forward-pass
            loss.backward()  # backward-pass
            opt.step() # update weights
            # calculate loss to show the user
            avg_loss += loss / len(data_tr)

        print('loss: %f' % avg_loss)

        # show intermediate results
        model.eval()
        Y_hat = model(X_val.to(device)).detach().cpu() # detach and put into cpu
        val_score = iou_pytorch(Y_val, Y_hat > 0.5).mean()
        val_loss = loss_fn(Y_val, Y_hat)

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_score'].append(val_score.item())

        # Save masks for the current epoch
        save_masks(epoch, model, data_tr, "train_masks")
        save_masks(epoch, model, data_val, "valid_masks")

    plot_history(history)

def score_model(model, metric, data):
    model.eval()  # testing mode
    scores = 0
    for X_batch, Y_label in data:
        Y_pred = (model(X_batch.to(device)) > 0.5).int()
        scores += metric(Y_pred, Y_label.to(device)).mean().item()
    return scores/len(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNet().to(device)
optimizer = optim.Adam(unet_model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

train(unet_model, optimizer, loss_fn, 5, train_loader, val_loader)
score = score_model(unet_model, iou_pytorch, val_loader)
print("Validation IOU:", score)
