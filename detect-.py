import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import cv2
import numpy as np

# Custom dataset class with additional checks and logging
class BoxDataset(Dataset):
    def __init__(self, color_dir, depth_dir, mask_dir, transform=None):
        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.color_files = sorted(os.listdir(color_dir))
        self.depth_files = sorted(os.listdir(depth_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
        assert len(self.color_files) == len(self.depth_files) == len(self.mask_files), "Mismatch in number of files"
        assert all(c == d == m for c, d, m in zip(self.color_files, self.depth_files, self.mask_files)), "Mismatch in filenames"

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        color_path = os.path.join(self.color_dir, self.color_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        try:
            color_image = Image.open(color_path).convert('RGB')
            depth_image = Image.open(depth_path).convert('I;16')
            mask_image = Image.open(mask_path).convert('L')
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            raise
        
        if self.transform:
            try:
                color_image = self.transform(color_image)
                depth_image = self.transform(depth_image)
                mask_image = self.transform(mask_image)
            except Exception as e:
                print(f"Error transforming image {idx}: {e}")
                raise
        
        return color_image, depth_image, mask_image

# Model definition
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            identity = self.shortcut(identity)
        out += identity
        return nn.ReLU(inplace=True)(out)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            for inputs, depths, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                depths = depths.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(torch.cat((inputs, depths), dim=1))
                    loss = criterion(outputs, masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'Epoch {epoch}/{num_epochs - 1}, {phase} Loss: {epoch_loss:.4f}')
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(n_channels=4, n_classes=1).to(device)
#model.load_state_dict(torch.load('box_detector_v2.pth'))
#model.load_state_dict(torch.load('box_detector7m3-80.pth'))
model.load_state_dict(torch.load('box_detector7-1-80.pth'))
model.eval()

def prepare_image(color_image_path, depth_image_path, transform):
    color_image = Image.open(color_image_path).convert('RGB')
    depth_image = Image.open(depth_image_path).convert('I;16')

    color_image = transform(color_image).unsqueeze(0)  # Добавляем batch dimension
    depth_image = transform(depth_image).unsqueeze(0)  # Добавляем batch dimension

    return color_image, depth_image

transform = transforms.Compose([
    transforms.ToTensor()
])

color_image_path = 'depth_004_Color.png'
depth_image_path = 'depth_004_Depth.png'

color_image, depth_image = prepare_image(color_image_path, depth_image_path, transform)

with torch.no_grad():
    inputs = torch.cat((color_image, depth_image), dim=1).to(device) # Объединение цветного и глубинного изображений по каналу
    output = model(inputs)
    output = torch.sigmoid(output).cpu().numpy()

output_image = (output[0, 0] > 0.5).astype(np.uint8) # Бинаризация
contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

color_image = cv2.imread(color_image_path)
cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)

cv2.imwrite('detected_boxes_exact6-7-1-80-004-t0.5.png', color_image)
cv2.imshow('Detected Boxes', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
