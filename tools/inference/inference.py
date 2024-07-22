import time
import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# 添加项目根目录到 PYTHONPATH
project_path = 'D:\\pythondata\\gitYOLO\\yolov9-main'  # 替换为你的项目路径
sys.path.append(os.path.abspath(project_path))

from models.yolo import Model  # 假设模型结构在 models/yolo.py 中定义

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # 调整图像大小为 640x640
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载并预处理图像
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 加载图像并转换为 RGB
    image = preprocess(image)  # 预处理图像
    image = image.unsqueeze(0)  # 增加批次维度
    return image.to(device)  # 将图像移动到 GPU

# 加载模型权重文件
checkpoint_path_before = 'D:\\BaiduSyncdisk\\result\\visdrone\\v9c-noRep-vd200\\weights\\best.pt'
checkpoint_before = torch.load(checkpoint_path_before, map_location=device)

# 提取模型对象
model_before = checkpoint_before['model']
model_before.eval()

# 将模型权重转换为 float 类型
model_before = model_before.float().to(device)  # 将模型移动到 GPU

# 加载图像
image_path = 'D:\\pythondata\\gitYOLO\\yolov9-main\\data\\images\\0000205_02019_d_0000201.jpg'  # 替换为你的图像路径
input_data = load_image(image_path)

# 计时
start_time = time.time()
with torch.no_grad():
    for _ in range(200):  # 多次推理取平均
        output = model_before(input_data)
end_time = time.time()

time_before = (end_time - start_time) / 200
print(f'重参数化前模型平均推理时间: {time_before} 秒')

#############after

# 加载重参数化后的权重文件
checkpoint_path_after = 'D:\\BaiduSyncdisk\\result\\visdrone\\v9c_2RepC2fCIB_DC2_vd200\\weights\\2RepC2fCIB_DC2best_vd200.pt'
checkpoint_after = torch.load(checkpoint_path_after, map_location=device)

# 提取重参数化后的模型对象
model_after = checkpoint_after['model']
model_after.eval()

# 将模型权重转换为 float 类型
model_after = model_after.float().to(device)  # 将模型移动到 GPU

# 加载图像
input_data = load_image(image_path)

# 计时
start_time = time.time()
with torch.no_grad():
    for _ in range(200):  # 多次推理取平均
        output = model_after(input_data)
end_time = time.time()

time_after = (end_time - start_time) / 200
print(f'重参数化后模型平均推理时间: {time_after} 秒')
