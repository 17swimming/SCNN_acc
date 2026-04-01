import torch
# from core_copy import Core
from core_copy import Core
# 创建Core实例
core = Core(kernel_size=3, num_pus=2)

cin = 1
cout = 2
H = 6
W = 6
padding = 0
stride = 1
k = 3
oh = (H - k + 2 * padding) // stride + 1
ow = (W - k + 2 * padding) // stride + 1
# if[Cin,H,W]
if_tensor = torch.zeros(cin, H, W, dtype=torch.int32)

if_map = torch.tensor([
    [1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1]
], dtype=torch.int32)

for i in range(cin):
    if_tensor[i, :, :] = if_map
        
# 定义测试卷积核 [k, k]
kernel_cin = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.int32)

kernel = torch.zeros(cout, cin, k, k, dtype=torch.int32)
for i in range(cout):
    for j in range(cin):
        kernel[i, j, :, :] = kernel_cin

# 初始化输出
output = torch.zeros((cout, oh, ow), dtype=torch.float32)

# 每个cin调用一次core
for i in range(cin):
    core.configure_weights(i, kernel)
    # # 检查权重，对于W0，权重应该是321
    # print("W0 weights:", core.pus[0]['W0'].weights)
    # # 检查权重，对于W1，权重应该是654
    # print("W1 weights:", core.pus[0]['W1'].weights)
    # # 检查权重，对于W2，权重应该是987
    # print("W2 weights:", core.pus[0]['W2'].weights)

    h,w = if_map.shape

    # 不同cin之间的输出累加起来
    # output += core.process(if_tensor[i, :, :], kernel.shape[0], i, oh, ow)
    output,cycles,pe_cycles = core.process(if_map, kernel.shape[0] , i, oh, ow)
    print('w0的利用率为{}'.format(pe_cycles[0]/cycles))
    print('w1的利用率为{}'.format(pe_cycles[1]/cycles))
    print('w2的利用率为{}'.format(pe_cycles[2]/cycles))
# 使用PyTorch计算卷积结果进行比较
# 准备输入数据，形状为 [B, C, H, W]
input_tensor = if_map.unsqueeze(0).unsqueeze(0).float()
# 准备权重，形状为 [Cout, Cin, K, K]
weight_tensor = kernel.float()
# 执行卷积，不使用偏置
pytorch_output = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias=None, stride=1, padding=0)

# 调整输出形状以匹配PyTorch的输出形状 (B, C, H, W)
output = output.unsqueeze(0)

# 打印结果
# print("Input feature map:")
# print(if_map)
# print("\nOutput feature map shape:", output.shape)
# print("\nOutput feature map for Cout 0:")
# print(output[0, 0, :, :])

# print("\nPyTorch output shape:", pytorch_output.shape)
# print("\nPyTorch output for Cout 0:")
# print(pytorch_output[0, 0, :, :])

# 比较结果
print("\nComparing results...")
# 检查形状是否相同
if output.shape != pytorch_output.shape:
    print("❌ Shape mismatch!")
else:
    # 检查值是否相同（允许小的浮点误差）
    is_close = torch.allclose(output.float(), pytorch_output, atol=1e-6)
    if is_close:
        print("✅ Test passed! Results match PyTorch convolution.")
    else:
        print("❌ Test failed! Results don't match PyTorch convolution.")
        # 打印差异
        diff = torch.abs(output.float() - pytorch_output)
        print("\nDifference:")
        print(diff)
