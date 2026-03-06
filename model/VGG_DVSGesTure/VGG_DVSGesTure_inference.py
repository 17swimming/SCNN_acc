import torch
import numpy as np
import os
import sys
from VGG_models import VGGSNN
from spikingjelly.datasets import DVS128Gesture
from torch.utils.data import DataLoader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载模型
model_path = 'dvsgesture_T10_TET1_seed1000_cut1_best.pth'
model = VGGSNN(num_classes=11, img_width=48)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")

# 加载数据集
dataset = DVS128Gesture(root='../../data/DvsGesture', train=False, data_type='frame', frames_number=10, split_by='number')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f"Dataset loaded successfully. Test samples: {len(dataset)}")

# 创建保存目录
save_dir = '../../conv/VGG_DVSGesture'
os.makedirs(save_dir, exist_ok=True)

# 注册钩子来捕获卷积层的输入
conv_inputs = {}

def get_conv_input(name):
    def hook(model, input, output):
        # 只保存第一个 batch 的输入
        if 'batch' not in conv_inputs:
            conv_inputs['batch'] = 0
        if conv_inputs['batch'] == 0:
            # 输入是一个元组，第一个元素是输入张量
            # 将输入转换为 0/1 二值化
            input_tensor = input[0].detach().cpu().numpy()
            binary_input = (input_tensor > 0).astype(np.float32)
            # 将维度顺序从 B\T\C\H\W 转换为 T\B\C\H\W
            if binary_input.ndim == 5:
                binary_input = np.transpose(binary_input, (1, 0, 2, 3, 4))
            conv_inputs[name] = binary_input
            # 保存后将batch标记为1，避免后续batch覆盖
            conv_inputs['batch'] = 1
    return hook

# 遍历模型的所有模块，为卷积层注册钩子
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        module.register_forward_hook(get_conv_input(name))

# 进行推理
print("Starting inference...")
correct = 0
total = 0

# 该段代码的功能：
# 1. 在推理阶段禁用梯度计算，遍历测试数据加载器中的每个 batch。
# 2. 将每个 batch 的 DVS 帧数据 (128×128) 通过双线性插值缩放到 48×48，以满足模型输入尺寸要求。
# 3. 将缩放后的时空数据送入 VGGSNN 模型进行前向传播，得到每帧的 11 类 logits。
# 4. 对时间维度取平均后得到样本级预测，与真实标签比较并累计正确数，实时打印 batch 级准确率。
# 5. 推理完成后，把各卷积层首次遇到的二值化输入保存到本地，用于后续可视化或分析。

with torch.no_grad():
    for i, (data, label) in enumerate(dataloader):
        data = data.to(device)
        label = label.to(device)
        
        print(f"Processing batch {i}, data shape: {data.shape}, labels: {label}")
        
        # 调整输入大小从 128x128 到 48x48
        import torch.nn.functional as F
        # 合并 batch 和 time 维度，得到形状 [N*T, C, H, W]
        batch_size, time_steps, channels, height, width = data.shape
        data_reshaped = data.reshape(batch_size * time_steps, channels, height, width)
        # 进行插值
        data_resized = F.interpolate(data_reshaped, size=(48, 48), mode='bilinear', align_corners=False)
        # 恢复原始形状 [N, T, C, H, W]
        data = data_resized.reshape(batch_size, time_steps, channels, 48, 48)
        print(f"Resized data shape: {data.shape}")
        
        # 前向传播
        output = model(data)
        print(f"Inference completed. Output shape: {output.shape}")
        
        # 计算准确率：对时间维度取平均，然后取最大概率的类别
        output_aggregated = output.mean(dim=1)  # [batch_size, num_classes]
        _, predicted = torch.max(output_aggregated, 1)  # [batch_size]
        correct += (predicted == label).sum().item()
        total += label.size(0)
        print(f"Batch {i} accuracy: {(predicted == label).sum().item() / label.size(0):.4f}")

# 保存卷积层的输入
print("Saving convolution layer inputs...")
for name, input_data in conv_inputs.items():
    if name != 'batch':
        save_path = os.path.join(save_dir, f"{name}_input.npy")
        np.save(save_path, input_data)
        print(f"Saved {name} input to {save_path}")
        print(f"Shape: {input_data.shape}")

# 打印总体准确率
print(f"\nTotal accuracy: {correct / total * 100:.2f}%")
print(f"Correct predictions: {correct}, Total samples: {total}")

print("All done!")
