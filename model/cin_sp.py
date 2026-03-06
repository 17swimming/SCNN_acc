import numpy as np
import torch
import os

def main():
    # 定义B=2文件夹路径
    folder_path = '../conv/B=2'
    
    # 收集所有input.npy文件
    input_files = [f for f in os.listdir(folder_path) if f.endswith('input.npy')]
    
    # 准备结果存储
    results = []
    
    # 遍历所有input文件
    for file_name in input_files:
        print(f"Processing file: {file_name}")
        
        # 读取npy文件
        file_path = os.path.join(folder_path, file_name)
        input_tensor = np.load(file_path)
        input_tensor = torch.from_numpy(input_tensor)
        
        # 解析张量形状
        TB, Cin, h, w = input_tensor.shape
        T = 4  # 假设时间步长为4
        B = TB // T
        input_tensor = input_tensor.view(T, B, Cin, h, w)
        
        # 遍历每个cin通道，统计稀疏度
        cin_sparsity = []
        for cin in range(Cin):
            # 提取当前cin通道的所有数据
            cin_data = input_tensor[:, :, cin, :, :]
            # 计算稀疏度（零值比例）
            zero_count = (cin_data == 0).sum().item()
            total_count = cin_data.numel()
            sparsity = zero_count / total_count
            cin_sparsity.append(sparsity)
        
        # 计算平均稀疏度
        avg_sparsity = sum(cin_sparsity) / Cin
        
        # 保存结果
        results.append({
            'file_name': file_name,
            'cin_count': Cin,
            'avg_sparsity': avg_sparsity,
            'cin_sparsity': cin_sparsity
        })
    
    # 将结果保存到文件
    output_file = 'cin_sparsity_results.txt'
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"File: {result['file_name']}\n")
            f.write(f"Cin count: {result['cin_count']}\n")
            f.write(f"Average sparsity: {result['avg_sparsity']:.6f}\n")
            f.write("Cin sparsity per channel:\n")
            for i, sparsity in enumerate(result['cin_sparsity']):
                f.write(f"  Cin {i}: {sparsity:.6f}\n")
            f.write("\n")
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
