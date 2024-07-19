import time
import torch
from models.yolo import Model  # 假设模型结构在 models/yolo.py 中定义

# 加载权重文件
checkpoint = torch.load('D:\\BaiduSyncdisk\\result\\visdrone\\v9c-noRep-vd200\\weights\\best.pt')

# 提取模型对象
model_before = checkpoint['model']
model_before.eval()

# 将模型权重转换为 float 类型
model_before = model_before.float()

# 准备输入数据（假设输入大小为 640x640）
input_size = 640  # 根据实际模型配置调整
input_data = torch.randn(1, 3, input_size, input_size).float()  # 转换为 float tensor 类型

# 计时
start_time = time.time()
with torch.no_grad():
    for _ in range(50):  # 多次推理取平均
        output = model_before(input_data)
end_time = time.time()

time_before = (end_time - start_time) / 50  # 这里要除以 10 而不是 100
print(f'重参数化前模型平均推理时间: {time_before} 秒')

#############after

# 加载重参数化后的权重文件
checkpoint_after = torch.load('D:\\BaiduSyncdisk\\result\\visdrone\\v9c_2RepC2fCIB_DC2_vd200\\weights\\2RepC2fCIB_DC2best_vd200.pt')

# 提取重参数化后的模型对象
model_after = checkpoint_after['model']
model_after.eval()

# 将模型权重转换为 float 类型
model_after = model_after.float()

# 准备输入数据（假设输入大小为 640x640）
input_size = 640  # 根据实际模型配置调整
input_data = torch.randn(1, 3, input_size, input_size).float()  # 转换为 float tensor 类型

# 计时
start_time = time.time()
with torch.no_grad():
    for _ in range(50):  # 多次推理取平均
        output = model_after(input_data)
end_time = time.time()

time_after = (end_time - start_time) / 50  # 这里要除以 10 而不是 100
print(f'重参数化后模型平均推理时间: {time_after} 秒')
