import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import numpy as np


# Define Conv and RepConv classes (simplified versions)
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class DualConv2(nn.Module):
 
    def __init__(self, in_channels, out_channels, stride=1, g=2):
        """
        Initialize the DualConv class.
        
        :param in_channels: int, the number of input channels
        :param out_channels: int, the number of output channels
        :param stride: int, convolution stride, default is 1
        :param g: int, the number of groups for the group convolution, default is 2
        """
        super().__init__()

        if in_channels % g != 0:
            raise ValueError("in_channels must be divisible by the number of groups (g).")
        if out_channels % g != 0:
            raise ValueError("out_channels must be divisible by the number of groups (g).")

        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.gc_bn = nn.BatchNorm2d(out_channels)
        
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.pwc_bn = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)

        # Flag to indicate whether to use reparameterized structure
        self.reparameterized = False

    def reparameterize(self):
        """
        Reparameterize the structure by merging the group convolution and pointwise convolution
        into a single convolutional layer.
        """
        with torch.no_grad():
            # Combine weights and biases of gc and pwc
            gc_weight = self.gc.weight
            pwc_weight = self.pwc.weight

            combined_weight = gc_weight + nn.functional.pad(pwc_weight, [1, 1, 1, 1])
            combined_bn_weight = self.gc_bn.weight + self.pwc_bn.weight
            combined_bn_bias = self.gc_bn.bias + self.pwc_bn.bias

            # Create a new convolutional layer with combined weights and biases
            self.reparam_conv = nn.Conv2d(self.gc.in_channels,
                                          self.gc.out_channels,
                                          kernel_size=3,
                                          stride=self.gc.stride,
                                          padding=1,
                                          bias=True)

            self.reparam_conv.weight = nn.Parameter(combined_weight)
            self.reparam_conv.bias = nn.Parameter(combined_bn_bias)

            # Remove original layers
            del self.gc
            del self.gc_bn
            del self.pwc
            del self.pwc_bn

            self.reparameterized = True

    def forward(self, x):
        """
        Define how DualConv processes the input images or input feature maps.
        
        :param x: torch.Tensor, input images or input feature maps
        :return: torch.Tensor, output feature maps
        """
        if self.reparameterized:
            out = self.reparam_conv(x)
        else:
            gc_out = self.gc_bn(self.gc(x))
            pwc_out = self.pwc_bn(self.pwc(x))
            out = gc_out + pwc_out
        
        return self.relu(out)

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
image_path = 'D:\\pythondata\\gitYOLO\\yolov9-main\\data\\images\\9999970_00000_d_0000012.jpg'  # Replace with your image path
image, original_size = load_image(image_path)

# Move model and image to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)

# Define the model
model = DualConv2(in_channels=3, out_channels=64, g=1).to(device)

# Forward pass
output = model(image)

# Base directory for saving feature maps
base_save_dir = 'D:\\pythondata\\gitYOLO\\yolov9-main\\runs\\DCdemo'

# Get the next available directory
save_dir = get_next_save_dir(base_save_dir)

# Save final output feature maps
save_feature_maps(output, 'Final_Output', save_dir, original_size, max_maps=3)
