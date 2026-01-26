import os
import math

# 设置环境变量以解决OpenMP库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from out_product import convolution_out_product
import torch
import time
import matplotlib.pyplot as plt
from simulator import OutProductSimulator

def main():
    # 使用一个for循环，不断增加dense的值，看加速比和dense有没有关系
    acc_list = []
    dense_list = []
    use_ratio_list = []
    Cin = 128  # 固定Cin值
    for dense in np.arange(0.01, 0.31, 0.01):  # 使用np.arange处理浮点数范围
        dense_list.append(dense)
        # print(f"------dense: {dense}---------")
        # 生成简单的输入张量和卷积核张量，输入张量里的值为0/1，卷积核张量里的值为固定顺序
        T, Batch, in_h, in_w = 2, 2, 8, 8
        Cout = Cin * 2

        # 直接使用PyTorch创建稀疏输入张量：只有几个位置为1，其余为0
        input_tensor = torch.zeros(T, Batch, Cin, in_h, in_w, dtype=torch.int8)

        # 在输入张量中随机设置一些1的位置
        torch.manual_seed(42)  # 设置随机种子以确保结果可复现
        num_ones = int(T * Batch * Cin * in_h * in_w * dense)  
        indices = torch.randperm(T*Batch*Cin*in_h*in_w)[:num_ones]
        input_tensor.view(-1)[indices] = 1

        # 直接使用PyTorch创建卷积核张量：所有输出通道和输入通道的卷积核都使用0-8的顺序值
        kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)

        # 输出特征图和输入特征图一样大
        total_add_num_inner_product =  in_h * in_w * Cin * 9 * Cout * T * Batch
        # convolution_result_out_product,total_add_num_out_product,total_cycle_num_out_product,_ = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
        #估计latency
        # #我的硬件参数
        # pe_num = 4           #一个PE里计算oh=ow=4，4个pe就是64
        # core_num= 8          #那就是一次可以计算8个Cout
        # total_cycle_num_out_product 是按照一个PE串行处理得到的值，现在使用使用4个pe，8个core，会快32倍
        # latency_estimation_out_product = total_cycle_num_out_product/pe_num/core_num
        
        # 使用模拟器完整计算
        simulator = OutProductSimulator()
        simulator.print_core_info = False
        output_tensor_sim, global_stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)
        # nvdla_large 的硬件参数
        Atomic_c = 64        #一次可以计算64个Cin
        nvdla_mac_num = 16   #那就是一次可以计算16个Cout
        # 稠密计算，由于Cin的最小单位是64，所以对Cin需要向上取整; Cout也是
        latency_estimation_nvdla = in_h * in_w  * 9  * T * Batch * math.ceil(Cout/nvdla_mac_num) * math.ceil(Cin/Atomic_c) 
        
        # 模拟中累加单元的使用率
        use_ratio = global_stats.num_ops / (global_stats.compute_cycles * 64)
        # 打印使用率
        print("总计算量：", global_stats.num_ops)
        print("总计算周期:", global_stats.total_cycles)
        print(f"dense: {dense}, use_ratio: {use_ratio:.4f}")
        use_ratio_list.append(use_ratio)

        acc_list.append(latency_estimation_nvdla/global_stats.total_cycles)
        # 如果加速比已经接近1，则结束循环
        if acc_list[-1] < 1.01:
            print(f"dense: {dense}, acc_ratio: {acc_list[-1]:.4f}")
            break
    
    print("平均利用率：", (sum(use_ratio_list)/len(use_ratio_list))*100)
    # 将acc_list和use_ratio_list使用折线图画出来,并保存图片
    plt.figure(figsize=(10, 6))
    # 绘制加速比曲线
    plt.plot(dense_list, acc_list, label='acc_ratio')
    # 绘制使用率曲线
    plt.plot(dense_list, use_ratio_list, label='use_ratio')
    # 添加加速比为1的红线
    plt.axhline(y=1, color='r', linestyle='-', label='acc_ratio=1')
    plt.xlabel("dense")
    plt.ylabel("value")
    plt.title("relation ship between metrics and dense")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 先保存图片，再显示
    plt.savefig("acc_ratio_vs_dense.png")
    plt.show()


if __name__ == '__main__':
    main()