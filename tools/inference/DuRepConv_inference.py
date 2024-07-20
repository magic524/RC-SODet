import time
import torch
import torch.nn as nn

import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class DualConv(nn.Module):
 
    def __init__(self, in_channels, out_channels, stride, g=2):
        """
        Initialize the DualConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in DualConv
        """
        super(DualConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
 
    def forward(self, input_data):
        """
        Define how DualConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data) + self.pwc(input_data)

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

####
def evaluate_model(model, input_size, iterations=50):
    model.eval()
    total_time = 0
    with torch.no_grad():
        for i in range(1, iterations + 1):
            start_time = time.time()
            input_data = torch.randn(1, 128, input_size, input_size).float().cuda()
            output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            iter_time = end_time - start_time
            total_time += iter_time
            if i % 10 == 0:
                avg_time = total_time / i
                print(f'Iteration {i} - 当前平均推理时间: {avg_time:.4f} 秒')
    return total_time / iterations

# 准备输入数据大小
input_size = 640

# 初始化模型并移动到GPU
model_dual_conv = DualConv(128, 128, stride=1, g=2).cuda()
model_dual_conv2 = DualConv2(128, 128, stride=1, g=2).cuda()

# 评估模型
time_dual_conv = evaluate_model(model_dual_conv, input_size)
print(f'DualConv 模型平均推理时间: {time_dual_conv:.4f} 秒')

time_dual_conv2 = evaluate_model(model_dual_conv2, input_size)
print(f'DualConv2 模型平均推理时间: {time_dual_conv2:.4f} 秒')

# 重参数化并评估 DualConv2 模型
model_dual_conv2.reparameterize()
time_dual_conv2_reparam = evaluate_model(model_dual_conv2, input_size)
print(f'Reparameterized DualConv2 模型平均推理时间: {time_dual_conv2_reparam:.4f} 秒')