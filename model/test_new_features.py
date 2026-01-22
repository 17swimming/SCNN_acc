import torch
import numpy as np
from out_product_simulator import OutProductSimulator
from out_product import convolution_out_product

def test_5d_input_4d_weight():
    """测试5D输入张量和4D权重张量的处理"""
    print("=== 测试5D输入张量和4D权重张量 ===")
    
    # 简化测试：使用T=1, Batch=1, Cin=1, 小尺寸输入
    T, Batch, Cin, in_h, in_w = 1, 1, 1, 4, 4
    Cout = 1  # 只测试一个输出通道
    
    # 创建稀疏输入张量：只有几个位置为1，其余为0
    input_tensor = torch.zeros(T, Batch, Cin, in_h, in_w, dtype=torch.float32)
    
    # 设置一个简单的输入模式
    input_tensor[0, 0, 0, 1, 1] = 1  # 中心位置为1
    
    # 创建卷积核张量：使用0-8的顺序值
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)
    
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"权重张量形状: {kernel_tensor.shape}")
    print("输入张量:")
    print(input_tensor.squeeze())
    print("卷积核:")
    print(kernel_tensor.squeeze())
    
    # 使用模拟器执行卷积
    simulator = OutProductSimulator()
    output_simulator, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)
    
    print(f"模拟器输出形状: {output_simulator.shape}")
    print(f"总操作数: {stats.num_ops}")
    print(f"总周期数: {stats.total_cycles}")
    print(f"计算周期数: {stats.compute_cycles}")
    print("模拟器输出:")
    print(output_simulator.squeeze())
    
    # 使用原始out_product.py验证结果
    output_original, total_add_num_original, total_cycle_num_original = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
    
    print(f"原始实现输出形状: {output_original.shape}")
    print(f"原始实现总操作数: {total_add_num_original}")
    print(f"原始实现计算周期数: {total_cycle_num_original}")
    print("原始实现输出:")
    print(output_original.squeeze())
    
    # 验证结果是否一致（考虑浮点精度误差）
    is_close = torch.allclose(output_simulator, output_original, rtol=1e-5)
    print(f"模拟器与原始实现结果是否一致: {is_close}")
    
    if not is_close:
        print("输出差异:")
        print(output_simulator.squeeze() - output_original.squeeze())
    
    return is_close

def test_pe_reuse():
    """测试PE复用功能"""
    print("\n=== 测试PE复用功能 ===")
    
    # 创建一个会产生大量tile的输入张量（确保tile数目远大于PE数目）
    T, Batch, Cin, in_h, in_w = 1, 1, 2, 16, 16  # 16x16输入会产生约4x4=16个tile，大于4个PE
    Cout = 1  # 只测试一个输出通道
    
    # 创建输入张量
    input_tensor = torch.zeros(T, Batch, Cin, in_h, in_w, dtype=torch.float32)
    torch.manual_seed(42)
    num_ones = int(T * Batch * Cin * in_h * in_w * 0.1)  # 10%的稀疏度
    indices = torch.randperm(T*Batch*Cin*in_h*in_w)[:num_ones]
    input_tensor.view(-1)[indices] = 1
    
    # 创建卷积核张量
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)
    
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"权重张量形状: {kernel_tensor.shape}")
    
    # 使用模拟器执行卷积
    simulator = OutProductSimulator()
    output, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)
    
    print(f"模拟器输出形状: {output.shape}")
    print(f"总操作数: {stats.num_ops}")
    print(f"总周期数: {stats.total_cycles}")
    print(f"计算周期数: {stats.compute_cycles}")
    
    # 检查tile数量是否超过PE数量（4个PE）
    # 16x16输入，padding=1，stride=1，输出尺寸为16x16
    # 每个tile处理4x4输出，所以总tile数为 (16/4) x (16/4) = 16个tile
    print(f"预计tile数量: 16")
    print(f"PE数量: 4")
    print(f"是否需要PE复用: {'是' if 16 > 4 else '否'}")
    
    # 验证输出是否正确
    output_original, _, _ = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
    is_close = torch.allclose(output, output_original, rtol=1e-5)
    print(f"结果是否正确: {is_close}")
    
    return is_close

def test_larger_input():
    """测试更大的输入张量"""
    print("\n=== 测试更大的输入张量 ===")
    
    T, Batch, Cin, in_h, in_w = 1, 1, 4, 8, 8
    Cout = 1  # 只测试一个输出通道
    
    # 创建输入张量
    input_tensor = torch.zeros(T, Batch, Cin, in_h, in_w, dtype=torch.float32)
    torch.manual_seed(42)
    num_ones = int(T * Batch * Cin * in_h * in_w * 0.2)  # 20%的稀疏度
    indices = torch.randperm(T*Batch*Cin*in_h*in_w)[:num_ones]
    input_tensor.view(-1)[indices] = 1
    
    # 创建卷积核张量
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)
    
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"权重张量形状: {kernel_tensor.shape}")
    
    # 使用模拟器执行卷积
    simulator = OutProductSimulator()
    output, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)
    
    print(f"模拟器输出形状: {output.shape}")
    print(f"总操作数: {stats.num_ops}")
    print(f"总周期数: {stats.total_cycles}")
    print(f"计算周期数: {stats.compute_cycles}")
    
    # 验证结果
    output_original, _, _ = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
    is_close = torch.allclose(output, output_original, rtol=1e-5)
    print(f"结果是否正确: {is_close}")
    
    return is_close

if __name__ == "__main__":
    print("开始测试新功能...")
    
    # 运行所有测试
    tests = [
        test_5d_input_4d_weight,
        test_pe_reuse,
        test_larger_input
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试失败: {e}")
            results.append(False)
    
    # 汇总测试结果
    print("\n=== 测试结果汇总 ===")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "通过" if result else "失败"
        print(f"测试 {i+1}: {test.__name__} - {status}")
    
    if all(results):
        print("\n🎉 所有测试通过!")
    else:
        print("\n❌ 部分测试失败!")
