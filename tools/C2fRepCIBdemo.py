import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import numpy as np

# Define Conv and RepConv classes (simplified versions)
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1):
        super(RepConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RepCIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, padding=1, groups=c1),
            Conv(c1, 2 * c_, 1),
            RepConv(2 * c_, 2 * c_, 3, padding=1, groups=2 * c_) if not lk else RepConv(2 * c_, 2 * c_, 3, padding=1, groups=2 * c_),
            Conv(2 * c_, c2, 1),
            RepConv(c2, c2, 3, padding=1, groups=c2),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(RepCIB(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2fRepCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=2, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

# Load and preprocess an example image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image, original_size

def enhance_image(image_tensor, brightness_factor=4.0):
    image_array = image_tensor.detach().cpu().numpy()
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    image_array = np.uint8(255 * image_array)
    image_pil = Image.fromarray(image_array)
    
    # Adjust the brightness
    enhancer = ImageEnhance.Brightness(image_pil)
    enhanced_image = enhancer.enhance(brightness_factor)
    
    return enhanced_image

def save_feature_maps(feature_maps, layer_name, save_dir, input_size, max_maps=3):
    num_maps = min(feature_maps.size(1), max_maps)
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_maps):
        plt.figure(figsize=(input_size[0] / 100, input_size[1] / 100))
        feature_map_resized = transforms.functional.resize(
            enhance_image(feature_maps[0, i]), (input_size[1], input_size[0])
        )
        plt.imshow(feature_map_resized, cmap='viridis')
        plt.axis('off')
        save_path = os.path.join(save_dir, f"{layer_name}_feature_map_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# Find the next available directory for saving results
def get_next_save_dir(base_dir):
    index = 1
    while True:
        save_dir = os.path.join(base_dir, str(index))
        if not os.path.exists(save_dir):
            return save_dir
        index += 1

# Example usage
image_path = 'D:\\pythondata\\gitYOLO\\yolov9-main\\data\\images\\0000153_00001_d_0000001.jpg'  # Replace with your image path
image, original_size = load_image(image_path)

# Move model and image to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)

# Define the model
model = C2fRepCIB(c1=3, c2=64, shortcut=True, e=0.5, lk=False).to(device)

# Forward pass
output = model(image)

# Base directory for saving feature maps
base_save_dir = 'D:\\pythondata\\gitYOLO\\yolov9-main\\runs\\C2fRepCIBdemo'

# Get the next available directory
save_dir = get_next_save_dir(base_save_dir)

# Save final output feature maps
save_feature_maps(output, 'Final_Output', save_dir, original_size, max_maps=3)
