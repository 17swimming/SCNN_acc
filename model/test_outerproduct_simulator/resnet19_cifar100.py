import os
import sys
# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines import run_sato_conv
# 设置环境变量以解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
from outer_product.simulator.simulator_pe_ratio import OutProductSimulator
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns
def test_calc():
    T = 6   
    B = 2
    
    # 创建测试数据集列表，包含每个测试用例的输入文件路径和对应的Cout值
    # 使用绝对路径或相对于当前文件的路径
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases = [
        # Layer 1 (Cout=128)
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer1_0_conv1_inputs.npy'), 'Cout': 128},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer1_0_conv2_inputs.npy'), 'Cout': 128},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer1_1_conv1_inputs.npy'), 'Cout': 128},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer1_1_conv2_inputs.npy'), 'Cout': 128},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer1_2_conv1_inputs.npy'), 'Cout': 128},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer1_2_conv2_inputs.npy'), 'Cout': 128},  
        # Layer 2 (Cout=256)
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer2_0_conv1_inputs.npy'), 'Cout': 256},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer2_0_conv2_inputs.npy'), 'Cout': 256},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer2_1_conv1_inputs.npy'), 'Cout': 256},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer2_1_conv2_inputs.npy'), 'Cout': 256},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer2_2_conv1_inputs.npy'), 'Cout': 256},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer2_2_conv2_inputs.npy'), 'Cout': 256},  
        # Layer 3 (Cout=512)
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer3_0_conv1_inputs.npy'), 'Cout': 512},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer3_0_conv2_inputs.npy'), 'Cout': 512},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer3_1_conv1_inputs.npy'), 'Cout': 512},  
        {'file': os.path.join(current_dir, '../../conv/resnet19_cifar100/module_layer3_1_conv2_inputs.npy'), 'Cout': 512},  
    ]
    
    # 准备存储结果的列表
    results = []
    ours_total_cycle_num = 0
    dla_total_cycle_num = 0
    sato_total_cycle_num = 0
    all_pe_cycles = []  # 存储所有测试用例的PE cycle数
    all_compute_cycles = []  # 存储所有测试用例的计算周期数
    
    # 遍历测试数据集
    for i, test_case in enumerate(test_cases):
        print(f"\n=== 测试用例 {i+1}/{len(test_cases)}: {test_case['file']} ===")
        
        # 加载输入张量
        input_tensor = np.load(test_case['file']) #TB合并了
        Cout = test_case['Cout']
        
        input_tensor = torch.from_numpy(input_tensor)
        TB, Cin, h, w = input_tensor.shape
        #和SATO比较,input_tensor.shape=[B,C,H,W]
        sato_stats = run_sato_conv(Cout, input_tensor)   
        input_tensor = input_tensor.view(T, B, Cin, h, w)
        
        # 统计input_tensor的稠密度
        total_elements = input_tensor.numel()
        nonzero_elements = torch.sum(input_tensor != 0).item()
        dense = nonzero_elements / total_elements
        print(f"input_tensor的稠密度为：{dense:.2f}")
        
        # 生成kernel_tensor
        kernel_tensor = torch.ones(Cout, Cin, 3, 3)
        
        # 使用模拟器完整计算
        padding = 1
        stride = 1
        num_cores = 8
        num_pus = 16
        simulator = OutProductSimulator(num_cores=num_cores,num_pus=num_pus)
        output_tensor, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=padding, stride=stride)
        print(f"该测试用例中hazard_num: {simulator.hazard_num}")
        pe_utilization = simulator.pe_cycles / stats.compute_cycles
        generate_pe_utilization_heatmap(pe_utilization,filename=f'{test_case["file"].split("/")[-1].split(".")[0]}_{i}_pe_utilization.png')
        
        # 统计bitstream平均长度
        avg_bitstream_length = np.mean(simulator.bitstream_lengths)/h
        print(f"平均bitstream长度: {avg_bitstream_length:.2f}")
        
        all_pe_cycles.append(simulator.pe_cycles)
        all_compute_cycles.append(torch.tensor(stats.compute_cycles, dtype=torch.float32))
        
        oh =( h - kernel_tensor.shape[2] + 2 * padding) // stride + 1
        ow = ( w - kernel_tensor.shape[3] + 2 * padding) // stride + 1

        # 理论估计计算周期数,就是逐行
        theorical_cycles = oh * (ow+2*padding) * T * B * math.ceil(Cout/num_pus) * math.ceil(Cin/num_cores)
        # 稠密计算
        latency_estimation_nvdla = h * w * 9  * T * B * math.ceil(Cout/16) * math.ceil(Cin/64) 

        # SATO
        sato_cycles = sato_stats.compute_cycles
        
        # 计算加速比
        speedup = latency_estimation_nvdla / stats.compute_cycles
        sato_speedup = sato_cycles / stats.compute_cycles
        
        dla_total_cycle_num += latency_estimation_nvdla
        ours_total_cycle_num += stats.compute_cycles
        sato_total_cycle_num += sato_cycles
        
        # 打印结果
        print(f"模拟器输入形状: {input_tensor.shape}")
        print(f"模拟器输出形状: {output_tensor.shape}")
        print(f"模拟器总周期: {stats.total_cycles}")
        print(f"模拟器计算周期: {stats.compute_cycles}")
        print(f"模拟器理论计算周期: {theorical_cycles}")
        print(f"DLA周期: {latency_estimation_nvdla}")
        print(f"DLA加速比：{speedup:.2f}")
        print(f"SATO周期: {sato_cycles}")
        print(f"SATO加速比：{sato_speedup:.2f}")

        
        # 打印数据量统计
        # print("\n=== 数据量统计 ===")
        # print(f"从DRAM搬运的MP数据量: {stats.data_moved['mp']} bits")
        # print(f"理论kernel数据量:{B * Cout * Cin * 3 * 3 * 16} bits")
        # print(f"从DRAM搬运的kernel数据量: {stats.data_moved['kernel']} bits")
        # print(f"从DRAM搬运的act数据量: {stats.data_moved['act']} bits")
        total_data = stats.data_moved['mp'] + stats.data_moved['kernel'] + stats.data_moved['act']
        # print(f"总数据量: {total_data} bits ({total_data / 8 / 1024:.2f} KB)")
        
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
            'dla_speedup': speedup,
            'sato_cycles': sato_cycles,
            'sato_speedup': sato_speedup,
            'total_data_bits': total_data,
            'total_data_kb': total_data / 8 / 1024,
            'avg_bitstream_length': avg_bitstream_length
        })

    # 计算平均加速比
    avg_speedup =  dla_total_cycle_num / ours_total_cycle_num
    avg_sato_speedup = sato_total_cycle_num / ours_total_cycle_num
    print(f"\n网络DLA加速比：{avg_speedup:.2f}")
    print(f"网络SATO加速比：{avg_sato_speedup:.2f}")
    
    # 计算总体平均PE利用率
    if all_pe_cycles:
        stacked_pe_utils = torch.stack(all_pe_cycles)  # [num_tests, num_cores, 3]
        stacked_compute_cycles = torch.stack(all_compute_cycles)  # [num_tests]
        # 扩展compute_cycles维度以便广播除法
        stacked_compute_cycles = stacked_compute_cycles.view(-1, 1, 1)  # [num_tests, 1, 1]
        overall_avg_pe_utilization = stacked_pe_utils / stacked_compute_cycles
        # 计算各维度平均值
        avg_by_pe = torch.mean(overall_avg_pe_utilization, dim=(0, 1))  # 对测试用例和cores取平均
        print(f"\n=== 总体平均PE利用率 ===")
        print(f"W0 PE利用率: {avg_by_pe[0]:.4f}")
        print(f"W1 PE利用率: {avg_by_pe[1]:.4f}")
        print(f"W2 PE利用率: {avg_by_pe[2]:.4f}")
        print(f"总体平均PE利用率: {torch.mean(avg_by_pe):.4f}")
    
    # 将结果保存为CSV文件，文件名包含加速比结果
    filename = f'../../output/simulator_outproduct/core{num_cores}_pu{num_pus}_dla{avg_speedup:.2f}_sato{avg_sato_speedup:.2f}_simulator_mapping_test_results.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['test_case', 'file', 'Cout', 'input_shape', 'dense', 'simulator_cycles', 'theorical_cycles', 'dla_cycles', 'dla_speedup', 'sato_cycles', 'sato_speedup', 'total_data_bits', 'total_data_kb', 'avg_bitstream_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n测试完成，结果已保存到 {filename} 文件中")
    
    # 绘制模拟器计算周期占理论计算周期的百分比图
    plot_cycle_ratio(results, num_cores, num_pus)
    # 绘制速度对比图（以DLA为基准）
    plot_speedup_comparison(results, num_cores, num_pus)


def plot_cycle_ratio(results, num_cores, num_pus):
    """
    绘制模拟器计算周期和理论计算周期的对比图
    横轴：测试用例，纵轴：周期数
    每个测试用例有两个柱子：左边是理论计算周期，右边是计算周期
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # 提取数据
    test_case_ids = [r['test_case'] for r in results]
    file_paths = [r['file'] for r in results]
    # 提取测试用例名字中的"layerx_x_convx"字段
    def extract_layer_info(file_path):
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        for i, part in enumerate(parts):
            if part.startswith('layer'):
                layer_parts = []
                for j in range(i, len(parts)):
                    layer_parts.append(parts[j])
                    if parts[j].startswith('conv'):
                        break
                return '_'.join(layer_parts)
        return file_name
    
    layer_ids = [extract_layer_info(fp) for fp in file_paths]
    theoretical_cycles = [r['theorical_cycles'] for r in results]
    simulator_cycles = [r['simulator_cycles'] for r in results]
    
    # 计算平均值
    avg_theoretical = np.mean(theoretical_cycles)
    avg_simulator = np.mean(simulator_cycles)
    
    test_case_ids.append('Avg')
    layer_ids.append('Avg')
    theoretical_cycles.append(avg_theoretical)
    simulator_cycles.append(avg_simulator)
    
    # 增大图形尺寸，适应更多测试用例
    plt.figure(figsize=(24, 10))
    x = np.arange(len(test_case_ids))
    width = 0.25      # 柱子宽度减小
    gap = 0.1         # 柱子之间的间隔增大
    
    bars1 = plt.bar(x - (width + gap)/2, theoretical_cycles, width, label='Theoretical Cycles', 
                    color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
    bars2 = plt.bar(x + (width + gap)/2, simulator_cycles, width, label='Simulator Cycles', 
                    color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1)
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        plt.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{int(height1):,}',
                ha='center', va='bottom', fontsize=10, rotation=0)
        percentage = (height2 / height1) * 100
        plt.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=10, rotation=0)
        reduction = 100 - percentage
        arrow_x = bar1.get_x() + bar1.get_width()
        arrow_y_start = height1
        arrow_y_end = height2
        right_bar_top = bar2.get_x() + bar2.get_width()/2., height2
        plt.annotate('', xy=right_bar_top, xytext=(arrow_x, arrow_y_start),
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=2.5, alpha=0.8))
        arrow_mid_y = (arrow_y_start + arrow_y_end) / 2
        arrow_mid_x = (arrow_x + right_bar_top[0]) / 2
        plt.text(arrow_mid_x, arrow_mid_y, f'-{reduction:.1f}%',
                ha='center', va='center', fontsize=14, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.8))
    
    plt.xlabel('Test Case', fontsize=14, fontweight='bold')
    plt.ylabel('Cycles (log scale)', fontsize=14, fontweight='bold')
    plt.title(f'Cycle Comparison: Theoretical vs Simulator (cores={num_cores}, PUs={num_pus})', 
              fontsize=15, fontweight='bold', pad=20)
    # 减小横轴标签字体并旋转45度，避免重叠
    plt.xticks(x, [f'{i}' for i in layer_ids], fontsize=12, rotation=45, ha='right')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.yscale('log')
    
    output_dir = '../../output/simulator_outproduct/'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/cycle_comparison_cores{num_cores}_pu{num_pus}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n已保存周期对比图到: {filename}")
    
    avg_theoretical = np.mean(theoretical_cycles)
    avg_simulator = np.mean(simulator_cycles)
    avg_ratio = avg_simulator / avg_theoretical * 100
    print(f"平均理论周期: {avg_theoretical:,.0f}")
    print(f"平均模拟器周期: {avg_simulator:,.0f}")
    print(f"平均占比: {avg_ratio:.2f}%")


def plot_speedup_comparison(results, num_cores, num_pus):
    """
    绘制速度对比图，以DLA为基准（DLA speedup=1）
    横轴：测试用例
    纵轴：speedup
    每个测试用例有四个柱子：DLA, SATO, 理论计算周期, 计算周期
    标注每个测试用例的稀疏度
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    test_case_ids = [r['test_case'] for r in results]
    file_paths = [r['file'] for r in results]
    def extract_layer_info(file_path):
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        for i, part in enumerate(parts):
            if part.startswith('layer'):
                layer_parts = []
                for j in range(i, len(parts)):
                    layer_parts.append(parts[j])
                    if parts[j].startswith('conv'):
                        break
                return '_'.join(layer_parts)
        return file_name
    
    layer_ids = [extract_layer_info(fp) for fp in file_paths]
    dense_values = [r['dense'] for r in results]
    
    dla_speedups = [1.0 for r in results]
    sato_speedups = [r['dla_cycles'] / r['sato_cycles'] for r in results]
    theoretical_speedups = [r['dla_cycles'] / r['theorical_cycles'] for r in results]
    simulator_speedups = [r['dla_speedup'] for r in results]
    
    avg_sato = np.mean(sato_speedups)
    avg_theoretical = np.mean(theoretical_speedups)
    avg_simulator = np.mean(simulator_speedups)
    
    test_case_ids.append('Avg')
    layer_ids.append('Avg')
    dense_values.append(0)
    dla_speedups.append(1.0)
    sato_speedups.append(avg_sato)
    theoretical_speedups.append(avg_theoretical)
    simulator_speedups.append(avg_simulator)
    
    # 增大图形尺寸
    plt.figure(figsize=(28, 10))
    x = np.arange(len(test_case_ids))
    width = 0.15      # 减小柱子宽度
    gap = 0.05        # 增大组内间距
    
    bars1 = plt.bar(x - 1.5*width - 1.5*gap, dla_speedups, width, label='DLA', 
                    color='lightgray', alpha=0.8, edgecolor='gray', linewidth=1)
    bars2 = plt.bar(x - 0.5*width - 0.5*gap, sato_speedups, width, label='SATO', 
                    color='lightgreen', alpha=0.8, edgecolor='green', linewidth=1)
    bars3 = plt.bar(x + 0.5*width + 0.5*gap, theoretical_speedups, width, label='Theoretical', 
                    color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
    bars4 = plt.bar(x + 1.5*width + 1.5*gap, simulator_speedups, width, label='Simulator', 
                    color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    # 调整稀疏度标注位置，避免与柱子底部标签冲突
    for i, dense in enumerate(dense_values):
        if i < len(test_case_ids) - 1:
            plt.text(x[i], 0.05, f'Sparsity: {dense:.2f}',
                    ha='center', va='center', fontsize=9, color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.xlabel('Test Case', fontsize=14, fontweight='bold')
    plt.ylabel('Speedup (DLA=1)', fontsize=14, fontweight='bold')
    plt.title(f'Speedup Comparison: DLA vs SATO vs Theoretical vs Simulator (cores={num_cores}, PUs={num_pus})', 
              fontsize=15, fontweight='bold', pad=20)
    # 减小横轴标签字体并旋转45度
    plt.xticks(x, [f'{i}' for i in layer_ids], fontsize=12, rotation=45, ha='right')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.ylim(0, max(max(sato_speedups), max(theoretical_speedups), max(simulator_speedups)) * 1.2)
    
    output_dir = '../../output/simulator_outproduct/'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/speedup_comparison_cores{num_cores}_pu{num_pus}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n已保存速度对比图到: {filename}")
    
    print(f"平均SATO speedup: {avg_sato:.2f}")
    print(f"平均理论计算speedup: {avg_theoretical:.2f}")
    print(f"平均模拟器speedup: {avg_simulator:.2f}")
def generate_pe_utilization_heatmap(pe_utilization,filename='pe_utilization.png'):
        """
        生成PE利用率热图
        :param pe_utilization: 形状为[num_cores, 3]的tensor，存储每个core的三个PE的利用率
        """
        # 准备数据
        utilization_matrix = pe_utilization.numpy()
        num_cores = 8
        
        # 创建热图，设置颜色范围为0-1
        plt.figure(figsize=(10, 8))
        sns.heatmap(utilization_matrix, annot=True, cmap='YlGnBu', fmt='.2f', 
                    xticklabels=['W0', 'W1', 'W2'], 
                    yticklabels=[f'Core {i}' for i in range(num_cores)],
                    vmin=0, vmax=1)
        
        plt.title('PE Utilization Heatmap')
        plt.xlabel('PE Type')
        plt.ylabel('Core')
        
        # 保存热图
        output_dir = '../../output/simulator_outproduct/resnet_cifar100'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/{filename}'
        plt.savefig(filename)
        plt.close()
        print(f"Saved PE utilization heatmap to {filename}")

if __name__ == "__main__":
    test_calc()
