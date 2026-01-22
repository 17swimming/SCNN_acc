import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入原始实现和模拟器
from out_product_1 import convolution_with_padding_out_product, convolution_with_out_product
from out_product_simulator import OutProductSimulator, Core, PE

def test_pe_vs_original():
    """
    测试PE类与原始convolution_with_out_product函数的正确性比较
    """
    print("测试PE类与原始convolution_with_out_product函数的正确性比较...")
    
    # 创建6x6输入矩阵
    input_matrix = torch.zeros((6, 6), dtype=torch.float32)
    input_matrix[1, 1] = 1  # 中心位置
    input_matrix[2, 3] = 1  # 另一个位置
    
    # 创建3x3卷积核
    kernel = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    
    # 原始实现
    original_output, original_add_num, original_cycle_num = convolution_with_out_product(input_matrix, kernel)
    
    # PE模拟器
    pe = PE("test_pe", 0)
    pe_output, pe_add_num, pe_cycle_num, _ = pe.process(input_matrix.unsqueeze(0), kernel.unsqueeze(0))
    
    # 比较结果
    if torch.allclose(original_output, pe_output[0]):
        print("✓ PE输出与原始实现一致")
    else:
        print("✗ PE输出与原始实现不一致")
        print("原始输出:", original_output)
        print("PE输出:", pe_output[0])
    
    # 比较累加次数
    if original_add_num == pe_add_num:
        print("✓ PE累加次数与原始实现一致")
    else:
        print("✗ PE累加次数与原始实现不一致")
        print("原始累加次数:", original_add_num)
        print("PE累加次数:", pe_add_num)
    
    # 比较计算拍数
    if original_cycle_num == pe_cycle_num:
        print("✓ PE计算拍数与原始实现一致")
    else:
        print("✗ PE计算拍数与原始实现不一致")
        print("原始计算拍数:", original_cycle_num)
        print("PE计算拍数:", pe_cycle_num)
    
    print()

def test_core_vs_original():
    """
    测试Core类与原始convolution_with_padding_out_product函数的正确性比较
    """
    print("测试Core类与原始convolution_with_padding_out_product函数的正确性比较...")
    
    # 创建4x4输入矩阵
    input_matrix = torch.zeros((4, 4), dtype=torch.float32)
    input_matrix[0, 0] = 1
    input_matrix[0, 1] = 1
    input_matrix[1, 0] = 1
    input_matrix[1, 1] = 1
    
    # 创建3x3卷积核
    kernel = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    
    # 原始实现
    original_output, original_add_num, original_cycle_num, original_tile_num = convolution_with_padding_out_product(input_matrix, kernel)
    
    # Core模拟器
    core = Core("test_core", 0, num_pes=16)
    core_output, _ = core.process(input_matrix.unsqueeze(0), kernel.unsqueeze(0))
    
    # 比较结果
    if torch.allclose(original_output, core_output[0]):
        print("✓ Core输出与原始实现一致")
    else:
        print("✗ Core输出与原始实现不一致")
        print("原始输出:", original_output)
        print("Core输出:", core_output[0])
    
    # 比较累加次数
    if original_add_num == core.total_add_num:
        print("✓ Core累加次数与原始实现一致")
    else:
        print("✗ Core累加次数与原始实现不一致")
        print("原始累加次数:", original_add_num)
        print("Core累加次数:", core.total_add_num)
    
    # 比较分块数量
    if original_tile_num == core.total_tile_num:
        print("✓ Core分块数量与原始实现一致")
    else:
        print("✗ Core分块数量与原始实现不一致")
        print("原始分块数量:", original_tile_num)
        print("Core分块数量:", core.total_tile_num)
    
    print()

def test_simulator():
    """
    测试完整的OutProductSimulator
    """
    print("测试完整的OutProductSimulator...")
    
    # 创建4x4输入矩阵
    input_matrix = torch.zeros((4, 4), dtype=torch.float32)
    input_matrix[0, 0] = 1
    input_matrix[0, 1] = 1
    input_matrix[1, 0] = 1
    input_matrix[1, 1] = 1
    
    # 创建3x3卷积核
    kernel = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    
    # 创建并运行模拟器
    simulator = OutProductSimulator()
    # 将输入转换为5D张量 [T,B,Cin,H,W]
    input_5d = input_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # 将卷积核转换为4D张量 [Cout,Cin,k_h,k_w]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
    output, stats = simulator.run_convolution(input_5d, kernel_4d, padding=1, stride=1)
    
    print("模拟器运行成功!")
    print("总周期数:", stats.total_cycles)
    print("计算周期数:", stats.compute_cycles)
    print("总操作数:", stats.num_ops)
    print("输出:", output)
    print()

def test_larger_input():
    """
    测试更大的输入矩阵
    """
    print("测试更大的输入矩阵...")
    
    # 创建8x8输入矩阵
    input_matrix = torch.zeros((8, 8), dtype=torch.float32)
    # 在随机位置设置一些1
    torch.manual_seed(42)
    input_matrix.view(-1)[torch.randperm(64)[:10]] = 1
    
    # 创建3x3卷积核
    kernel = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    
    # 原始实现
    original_output, original_add_num, original_cycle_num, original_tile_num = convolution_with_padding_out_product(input_matrix, kernel)
    
    # 模拟器
    simulator = OutProductSimulator()
    # 将输入转换为5D张量 [T,B,Cin,H,W]
    input_5d = input_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # 将卷积核转换为4D张量 [Cout,Cin,k_h,k_w]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
    sim_output, stats = simulator.run_convolution(input_5d, kernel_4d)
    
    # 比较结果
    if torch.allclose(original_output, sim_output[0, 0, 0]):
        print("✓ 大输入下模拟器输出与原始实现一致")
    else:
        print("✗ 大输入下模拟器输出与原始实现不一致")
    
    print()

if __name__ == "__main__":
    test_pe_vs_original()
    test_core_vs_original()
    test_simulator()
    test_larger_input()
    print("所有测试完成!")
