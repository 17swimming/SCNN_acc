import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 计算两个二进制向量的Jaccard相似度
def hamming_similarity(cube1, cube2):
    """
    计算两个cube的Hamming相似度
    cube1, cube2: 形状为(Cin, 1, 1)的张量
    """
    # 将张量转换为numpy数组并展平
    cube1_flat = cube1.cpu().numpy().flatten()
    cube2_flat = cube2.cpu().numpy().flatten()
    
    # 计算Hamming距离（不同元素的个数）
    hamming_distance = np.sum(cube1_flat != cube2_flat)
    
    # 计算总元素个数
    total_elements = len(cube1_flat)
    
    # 计算Hamming相似度
    similarity = 1.0 - (hamming_distance / total_elements)
    
    return similarity


# 计算一个tile内9个cube的平均相似度
def calculate_tile_similarity(tile_cubes):
    """
    计算一个tile内9个cube的平均相似度
    
    Args:
        tile_cubes: 包含9个cube的列表，每个cube的形状为 (Cin, 1, 1)
        
    Returns:
        平均相似度值
    """
    if len(tile_cubes) != 9:
        raise ValueError("tile_cubes must contain exactly 9 cubes")
    
    total_similarity = 0
    num_pairs = 0
    
    # 计算所有两两组合的相似度
    for i in range(len(tile_cubes)):
        for j in range(i + 1, len(tile_cubes)):
            similarity = hamming_similarity(tile_cubes[i], tile_cubes[j])
            total_similarity += similarity
            num_pairs += 1
    
    # 计算平均相似度
    if num_pairs > 0:
        return total_similarity / num_pairs
    else:
        return 0.0


def test_calc():
    T = 6   
    B = 2
    
    # 创建测试数据集列表，包含每个测试用例的输入文件路径和对应的Cout值
    test_cases = [
        # Layer 1 (Cout=128)
        {'file': '../conv/resnet19_cifar100/module_layer1_0_conv1_inputs.npy', 'Cout': 128},  
        {'file': '../conv/resnet19_cifar100/module_layer1_0_conv2_inputs.npy', 'Cout': 128},  
        {'file': '../conv/resnet19_cifar100/module_layer1_1_conv1_inputs.npy', 'Cout': 128},  
        {'file': '../conv/resnet19_cifar100/module_layer1_1_conv2_inputs.npy', 'Cout': 128},  
        {'file': '../conv/resnet19_cifar100/module_layer1_2_conv1_inputs.npy', 'Cout': 128},  
        {'file': '../conv/resnet19_cifar100/module_layer1_2_conv2_inputs.npy', 'Cout': 128},  
        # Layer 2 (Cout=256)
        {'file': '../conv/resnet19_cifar100/module_layer2_0_conv1_inputs.npy', 'Cout': 256},  
        {'file': '../conv/resnet19_cifar100/module_layer2_0_conv2_inputs.npy', 'Cout': 256},  
        {'file': '../conv/resnet19_cifar100/module_layer2_1_conv1_inputs.npy', 'Cout': 256},  
        {'file': '../conv/resnet19_cifar100/module_layer2_1_conv2_inputs.npy', 'Cout': 256},  
        {'file': '../conv/resnet19_cifar100/module_layer2_2_conv1_inputs.npy', 'Cout': 256},  
        {'file': '../conv/resnet19_cifar100/module_layer2_2_conv2_inputs.npy', 'Cout': 256},  
        # Layer 3 (Cout=512)
        {'file': '../conv/resnet19_cifar100/module_layer3_0_conv1_inputs.npy', 'Cout': 512},  
        {'file': '../conv/resnet19_cifar100/module_layer3_0_conv2_inputs.npy', 'Cout': 512},  
        {'file': '../conv/resnet19_cifar100/module_layer3_1_conv1_inputs.npy', 'Cout': 512},  
        {'file': '../conv/resnet19_cifar100/module_layer3_1_conv2_inputs.npy', 'Cout': 512},  
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
        
        # 计算每个batch的cube相似度
        batch_similarities = []
        
        for b in range(B):
            tile_time_similarities = []
            
            # 遍历所有可能的tile位置（每个tile对应输出特征图上的一个位置）
            # 对于3x3卷积核，输入特征图上的每个3x3区域对应一个tile
            for tile_h in range(h - 2):  # 输入特征图上的tile起始行
                for tile_w in range(w - 2):  # 输入特征图上的tile起始列
                    # 收集所有时间步中该tile内的cube相似度
                    time_similarities = []
                    
                    # 遍历所有时间步
                    for t in range(T):
                        # 收集该时间步下该tile内的9个cube
                        tile_cubes = []
                        for dh in range(3):  #  tile内的行偏移
                            for dw in range(3):  #  tile内的列偏移
                                # 获取当前时间步的cube (Cin, 1, 1)
                                cube = input_tensor[t, b, :, tile_h + dh, tile_w + dw].unsqueeze(1).unsqueeze(2)
                                tile_cubes.append(cube)
                        
                        # 计算该时间步下该tile的相似度
                        if len(tile_cubes) == 9:
                            time_similarity = calculate_tile_similarity(tile_cubes)
                            time_similarities.append(time_similarity)
                    
                    # 计算该tile在所有时间步上的平均相似度
                    if time_similarities:
                        tile_avg_similarity = np.mean(time_similarities)
                        tile_time_similarities.append(tile_avg_similarity)
            
            # 计算当前batch的平均相似度（所有tile的平均值）
            if tile_time_similarities:
                batch_similarity = np.mean(tile_time_similarities)
                batch_similarities.append(batch_similarity)
                print(f"Batch {b+1} cube similarity: {batch_similarity:.4f}")
        
        # 计算所有batch的平均相似度
        if batch_similarities:
            avg_similarity = np.mean(batch_similarities)
            print(f"Average cube similarity across all batches: {avg_similarity:.4f}")
        
        # 保存结果
        results.append({
            'test_case': i+1,
            'file': test_case['file'],
            'Cout': Cout,
            'batch_similarities': batch_similarities,
            'avg_similarity': avg_similarity if batch_similarities else 0.0
        })
    
    # 打印最终结果
    print("\n=== 最终结果 ===")
    for result in results:
        print(f"Test case {result['test_case']} ({result['file']}):")
        print(f"  Cout: {result['Cout']}")
        print(f"  Batch similarities: {[f'{s:.4f}' for s in result['batch_similarities']]}")
        print(f"  Average similarity: {result['avg_similarity']:.4f}")
    
    return results


def plot_similarity(results):
    """
    将相似度结果绘制成折线图
    
    Args:
        results: 包含测试结果的列表
    """
    # 提取测试用例序号和平均相似度
    test_case_ids = [result['test_case'] for result in results]
    avg_similarities = [result['avg_similarity'] for result in results]
    
    # 创建折线图
    plt.figure(figsize=(12, 6))
    plt.plot(test_case_ids, avg_similarities, marker='o', linestyle='-', color='b', label='Average Similarity')
    
    # 设置图表标题和标签
    plt.title('Cube Similarity Across Test Cases', fontsize=16)
    plt.xlabel('Test Case ID', fontsize=12)
    plt.ylabel('Hamming Similarity', fontsize=12)
    
    # 设置x轴刻度
    plt.xticks(test_case_ids)
    
    # 设置y轴范围
    plt.ylim(0.7, 1.0)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend()
    
    # 保存图表
    plt.savefig('cube_similarity_plot.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()


if __name__ == "__main__":
    results = test_calc()
    plot_similarity(results)

