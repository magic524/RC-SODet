import time
import torch
import torch.nn as nn

import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.block.v10.block import C2f, RepVGGDW
from models.block.v10.conv import Conv, RepConv

#重参数化前的CIB实现
class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

# 重参数化后的 CIB 实现
class RepCIB(nn.Module): 
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepConv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            RepConv(c2, c2, 3, g=c2),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
    
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fRepCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=2, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

def evaluate_model(model, input_data, iterations=30):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(input_data)
            torch.cuda.synchronize()  # 确保GPU计算完成
    end_time = time.time()
    return (end_time - start_time) / iterations

# 准备输入数据
input_size = 640  # 根据实际模型配置调整
input_data = torch.randn(1, 3, input_size, input_size).float().cuda()  # 确保输入数据在GPU上

# 初始化模型并移动到GPU
model_cib = CIB(3, 3, shortcut=True, e=0.5).cuda()
model_c2f_cib = C2fCIB(3, 3, n=2, shortcut=False, lk=False, g=1, e=0.5).cuda()

model_rep_cib = RepCIB(3, 3, shortcut=True, e=0.5).cuda()
model_c2f_rep_cib = C2fRepCIB(3, 3, n=2, shortcut=False, lk=False, g=1, e=0.5).cuda()

# 评估模型
time_cib = evaluate_model(model_cib, input_data)
time_c2f_cib = evaluate_model(model_c2f_cib, input_data)

time_rep_cib = evaluate_model(model_rep_cib, input_data)
time_c2f_rep_cib = evaluate_model(model_c2f_rep_cib, input_data)

print(f'CIB 模型平均推理时间: {time_cib} 秒')
print(f'C2fCIB 模型平均推理时间: {time_c2f_cib} 秒')

print(f'RepCIB 模型平均推理时间: {time_rep_cib} 秒')
print(f'C2fRepCIB 模型平均推理时间: {time_c2f_rep_cib} 秒')

