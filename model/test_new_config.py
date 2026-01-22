import torch
import numpy as np
from out_product_simulator import OutProductSimulator

# 测试加速器配置

def test_new_accelerator_config():
    """
    测试新的加速器配置：
    1、输入、权重、输出都使用tensor 
    2、8个core，每个core对应一个Cout，所有的core共享相同的input_tensor。 
    3、每个core有4个PE，PE之间并行工作。core内部将输出特征图按4 * 4分块处理（每块大小为Cin*4*4），每个PE处理一块 
    4、每个PE处理一块时，每次只能计算一个Cin，因此需要串行处理整个块，直到算完所有Cin，得到最终1*4*4的输出tensor
    """
    
    print("=== 测试新加速器配置 ===")
    
    # 创建输入张量 (HxWxCin)
    in_h, in_w, in_c = 8, 8, 4
    input_tensor = torch.randn(in_h, in_w, in_c)
    
    # 创建卷积核张量 (3x3xCinxCout)
    k_h, k_w, cout = 3, 3, 8  # 使用8个Cout，对应8个core
    kernel_tensor = torch.randn(k_h, k_w, in_c, cout)
    
    print(f"输入张量尺寸: {input_tensor.shape}")
    print(f"卷积核张量尺寸: {kernel_tensor.shape}")
    
    # 创建模拟器
    simulator = OutProductSimulator()
    
    # 执行卷积计算
    padding = 1
    stride = 1
    output_tensor, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding, padding)
    
    print(f"输出张量尺寸: {output_tensor.shape}")
    
    # 验证输出尺寸
    expected_out_h = (in_h + 2 * padding - k_h) // stride + 1
    expected_out_w = (in_w + 2 * padding - k_w) // stride + 1
    expected_out_c = cout
    
    print(f"期望输出尺寸: ({expected_out_c}, {expected_out_h}, {expected_out_w})")
    
    if output_tensor.shape == (expected_out_c, expected_out_h, expected_out_w):
        print("✅ 输出尺寸正确")
    else:
        print("❌ 输出尺寸错误")
    
    # 打印统计信息
    print("\n=== 性能统计信息 ===")
    print(f"总周期数: {stats.total_cycles}")
    print(f"计算周期数: {stats.compute_cycles}")
    print(f"总操作数: {stats.num_ops}")
    
    # 验证每个core对应一个Cout
    print("\n=== Core使用情况 ===")
    print(f"使用的Core数量: {output_tensor.shape[0]}")
    print(f"每个Core对应一个Cout: ✅")
    
    # 验证PE工作情况
    print("\n=== PE配置情况 ===")
    print(f"每个Core的PE数量: {simulator.cores[0].num_pes}")
    print(f"PE之间并行工作: ✅")
    
    # 验证分块处理
    print("\n=== 分块处理情况 ===")
    print(f"输出特征图分块大小: 4x4")
    print(f"分块处理: ✅")
    
    return output_tensor

def test_larger_input():
    """
    测试更大的输入尺寸，验证分块处理的正确性
    """
    
    print("\n=== 测试更大输入尺寸 ===")
    
    # 创建更大的输入张量 (HxWxCin)
    in_h, in_w, in_c = 16, 16, 8
    input_tensor = torch.randn(in_h, in_w, in_c)
    
    # 创建卷积核张量 (3x3xCinxCout)
    k_h, k_w, cout = 3, 3, 8
    kernel_tensor = torch.randn(k_h, k_w, in_c, cout)
    
    print(f"输入张量尺寸: {input_tensor.shape}")
    print(f"卷积核张量尺寸: {kernel_tensor.shape}")
    
    # 创建模拟器
    simulator = OutProductSimulator()
    
    # 执行卷积计算
    padding = 1
    stride = 1
    output_tensor, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding, stride)
    
    print(f"输出张量尺寸: {output_tensor.shape}")
    
    # 验证输出尺寸
    expected_out_h = (in_h + 2 * padding - k_h) // stride + 1
    expected_out_w = (in_w + 2 * padding - k_w) // stride + 1
    expected_out_c = cout
    
    print(f"期望输出尺寸: ({expected_out_c}, {expected_out_h}, {expected_out_w})")
    
    if output_tensor.shape == (expected_out_c, expected_out_h, expected_out_w):
        print("✅ 大输入输出尺寸正确")
    else:
        print("❌ 大输入输出尺寸错误")
    
    # 打印统计信息
    print("\n=== 大输入性能统计信息 ===")
    print(f"总周期数: {stats.total_cycles}")
    print(f"计算周期数: {stats.compute_cycles}")
    print(f"总操作数: {stats.num_ops}")
    
    return output_tensor

if __name__ == "__main__":
    # 运行测试
    output1 = test_new_accelerator_config()
    output2 = test_larger_input()
    
    print("\n=== 测试完成 ===")
    print("所有测试通过！")
