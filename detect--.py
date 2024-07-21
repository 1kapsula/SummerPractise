import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
import json

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
#model.load_state_dict(torch.load('box_detector7m3-80.pth'))
model.load_state_dict(torch.load('box_detector7-1-80.pth'))
#model.load_state_dict(torch.load('box_detector_v2.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def prepare_image(color_image_path, depth_image_path, transform):
    color_image = Image.open(color_image_path).convert('RGB')
    depth_image = Image.open(depth_image_path).convert('I;16')

    color_tensor = transform(color_image).unsqueeze(0)  # Добавляем batch dimension
    depth_tensor = transform(depth_image).unsqueeze(0)  # Добавляем batch dimension

    return torch.cat((color_tensor, depth_tensor), dim=1)
def draw_boxes_from_json(json_path, color_image_path, depth_image_path, model, transform):
    with open(json_path, 'r') as f:
        data = json.load(f)

    filename = os.path.basename(color_image_path)
    if filename not in data:
        print(f"{filename} not found in JSON file.")
        return

    regions = data[filename]["regions"]

    color_image = cv2.imread(color_image_path)
    if color_image is None:
        print(f"Failed to load color image: {color_image_path}")
        return

    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        print(f"Failed to load depth image: {depth_image_path}")
        return

    for region in regions:
        all_points_x = region["shape_attributes"]["all_points_x"]
        all_points_y = region["shape_attributes"]["all_points_y"]
        polygon = np.array([list(zip(all_points_x, all_points_y))], dtype=np.int32)

        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, polygon, 255)

        x, y, w, h = cv2.boundingRect(polygon)
        color_crop = color_image[y:y+h, x:x+w]
        depth_crop = depth_image[y:y+h, x:x+w]

        color_crop_path = 'temp_color_crop.png'
        depth_crop_path = 'temp_depth_crop.png'

        cv2.imwrite(color_crop_path, color_crop)
        cv2.imwrite(depth_crop_path, depth_crop)

        inputs = prepare_image(color_crop_path, depth_crop_path, transform).to(device)

        with torch.no_grad():
            output = model(inputs)
            output = torch.sigmoid(output).cpu().numpy()

        segmentation_threshold = 0.5
        output_image = (output[0, 0] > segmentation_threshold).astype(np.uint8)

        contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_image, [contour + (x, y) for contour in contours], -1, (0, 255, 0), 2)

    output_filename = 'detected_boxes-004f.png'
    cv2.imwrite(output_filename, color_image)
    cv2.imshow('Detected Boxes', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor()
])

# Paths to input data
json_path = 'via_project_15Jul2024_13h17m_json.json'
color_image_path = 'depth_004_Color.png'
depth_image_path = 'depth_004_Depth.png'

# Run function to draw boxes based on the model
draw_boxes_from_json(json_path, color_image_path, depth_image_path, model, transform)
