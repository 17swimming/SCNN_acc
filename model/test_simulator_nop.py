import os
# 设置环境变量以解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from inner_product.simulator_nop import InnerProductSimulator
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def test_calc():
    T = 6   
    B = 2
    
    # 创建测试数据集列表，包含每个测试用例的输入文件路径和对应的Cout值
    test_cases = [
        # Layer 1 (Cout=128)
        {'file': '../conv/resnet19_cifar100/module_layer1_0_conv1_inputs.npy', 'Cout': 128},  
        # {'file': '../conv/resnet19_cifar100/module_layer1_0_conv2_inputs.npy', 'Cout': 128},  
        # {'file': '../conv/resnet19_cifar100/module_layer1_1_conv1_inputs.npy', 'Cout': 128},  
        # {'file': '../conv/resnet19_cifar100/module_layer1_1_conv2_inputs.npy', 'Cout': 128},  
        # {'file': '../conv/resnet19_cifar100/module_layer1_2_conv1_inputs.npy', 'Cout': 128},  
        # {'file': '../conv/resnet19_cifar100/module_layer1_2_conv2_inputs.npy', 'Cout': 128},  
        # # Layer 2 (Cout=256)
        # {'file': '../conv/resnet19_cifar100/module_layer2_0_conv1_inputs.npy', 'Cout': 256},  
        # {'file': '../conv/resnet19_cifar100/module_layer2_0_conv2_inputs.npy', 'Cout': 256},  
        # {'file': '../conv/resnet19_cifar100/module_layer2_1_conv1_inputs.npy', 'Cout': 256},  
        # {'file': '../conv/resnet19_cifar100/module_layer2_1_conv2_inputs.npy', 'Cout': 256},  
        # {'file': '../conv/resnet19_cifar100/module_layer2_2_conv1_inputs.npy', 'Cout': 256},  
        # {'file': '../conv/resnet19_cifar100/module_layer2_2_conv2_inputs.npy', 'Cout': 256},  
        # # Layer 3 (Cout=512)
        # {'file': '../conv/resnet19_cifar100/module_layer3_0_conv1_inputs.npy', 'Cout': 512},  
        # {'file': '../conv/resnet19_cifar100/module_layer3_0_conv2_inputs.npy', 'Cout': 512},  
        # {'file': '../conv/resnet19_cifar100/module_layer3_1_conv1_inputs.npy', 'Cout': 512},  
        # {'file': '../conv/resnet19_cifar100/module_layer3_1_conv2_inputs.npy', 'Cout': 512},  
    ]
    
    # 准备存储结果的列表
    results = []
    ours_total_cycle_num = 0
    dla_total_cycle_num = 0
    
    # 遍历测试数据集
    for i, test_case in enumerate(test_cases):
        print(f"\n=== 测试用例 {i+1}/{len(test_cases)}: {test_case['file']} ===")
        
        # 加载输入张量
        input_tensor = np.load(test_case['file']) #TB合并了
        Cout = test_case['Cout']
        
        input_tensor = torch.from_numpy(input_tensor)
        TB, Cin, h, w = input_tensor.shape
        input_tensor = input_tensor.view(T, B, Cin, h, w)
        
        # 统计input_tensor的稠密度
        total_elements = input_tensor.numel()
        nonzero_elements = torch.sum(input_tensor != 0).item()
        dense = nonzero_elements / total_elements
        print(f"input_tensor的稠密度为：{dense:.2f}")
        
        # 生成kernel_tensor
        kernel_tensor = torch.ones(Cout, Cin, 3, 3)
        
        # 使用模拟器完整计算
        simulator = InnerProductSimulator()
        padding = 1
        stride = 1
        output_tensor, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=padding, stride=stride)
        
        oh =( h - kernel_tensor.shape[2] + 2 * padding) // stride + 1
        ow = ( w - kernel_tensor.shape[3] + 2 * padding) // stride + 1

        # 理论估计计算周期数
        theorical_cycles = oh * ow * T * B * math.ceil(Cout/16) * math.ceil(Cin/64) 
        # 稠密计算
        latency_estimation_nvdla = h * w * 9  * T * B * math.ceil(Cout/16) * math.ceil(Cin/64) 
        
        # 计算加速比
        speedup = latency_estimation_nvdla / stats.compute_cycles
        
        dla_total_cycle_num += latency_estimation_nvdla
        ours_total_cycle_num += stats.compute_cycles
        
        # 打印结果
        print(f"模拟器输入形状: {input_tensor.shape}")
        print(f"模拟器输出形状: {output_tensor.shape}")
        print(f"模拟器总周期: {stats.total_cycles}")
        print(f"模拟器计算周期: {stats.compute_cycles}")
        print(f"模拟器理论计算周期: {theorical_cycles}")
        print(f"dla的周期: {latency_estimation_nvdla}")
        print(f"加速比：{speedup:.2f}")
        
        # 打印数据量统计
        print("\n=== 数据量统计 ===")
        print(f"从DRAM搬运的MP数据量: {stats.data_moved['mp']} bits")
        print(f"理论kernel数据量:{B * Cout * Cin * 3 * 3 * 16} bits")
        print(f"从DRAM搬运的kernel数据量: {stats.data_moved['kernel']} bits")
        print(f"从DRAM搬运的act数据量: {stats.data_moved['act']} bits")
        total_data = stats.data_moved['mp'] + stats.data_moved['kernel'] + stats.data_moved['act']
        print(f"总数据量: {total_data} bits ({total_data / 8 / 1024:.2f} KB)")
        
        # 保存结果
        results.append({
            'test_case': i+1,
            'file': test_case['file'],
            'Cout': Cout,
            'input_shape': input_tensor.shape,
            'dense': dense,
            'simulator_cycles': stats.total_cycles,
            'theorical_cycles': theorical_cycles,
            'dla_cycles': latency_estimation_nvdla,
            'speedup': speedup,
            'total_data_bits': total_data,
            'total_data_kb': total_data / 8 / 1024
        })

    # 计算平均加速比
    avg_speedup =  dla_total_cycle_num / ours_total_cycle_num
    print(f"\n网络加速比：{avg_speedup:.2f}")
    
    # 将结果保存为CSV文件
    with open(f'../output/simulator_nop_test_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['test_case', 'file', 'Cout', 'input_shape', 'dense', 'simulator_cycles', 'theorical_cycles', 'dla_cycles', 'speedup', 'total_data_bits', 'total_data_kb']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n测试完成，结果已保存到 ../output/simulator_nop_test_results.csv 文件中")


if __name__ == "__main__":
    test_calc()
