import time
import torch
from models.yolo import Model  # 假设模型结构在 models/yolo.py 中定义

# 定义函数进行推理和计时
def evaluate_model(model, input_data):
    model.eval()
    model = model.float()  # 确保模型权重为 float 类型
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):  # 多次推理取平均
            output = model(input_data)
    end_time = time.time()
    return (end_time - start_time) / 10

# 加载权重文件
checkpoint_before = torch.load('D:\\BaiduSyncdisk\\result\\visdrone\\v9c-vd200\\weights\\best.pt')
model_before = checkpoint_before['model']

checkpoint_after = torch.load('D:\\BaiduSyncdisk\\result\\visdrone\\v9c-vd200\\weights\\best_reparam.pt')
model_after = checkpoint_after['model']

# 准备输入数据
input_size = 640  # 根据实际模型配置调整
input_data = torch.randn(1, 3, input_size, input_size).float()

# 评估模型
time_before = evaluate_model(model_before, input_data)
time_after = evaluate_model(model_after, input_data)

print(f'重参数化前模型平均推理时间: {time_before} 秒')
print(f'重参数化后模型平均推理时间: {time_after} 秒')
