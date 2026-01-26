import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math
from out_product import convolution_out_product


def main():
    start_time = time.time()
    #读取npy文件,lif1的输出，实际上是lif2的输入
    input_tensor = np.load('../conv/B=128/patch_embed_proj_conv2_input.npy')
    input_tensor = torch.from_numpy(input_tensor)
    T,B,Cin,h,w = input_tensor.shape
    print("输入tenor的稀疏度：", (input_tensor == 0).sum() / input_tensor.numel())

    output_tensor = np.load('../conv/B=128/patch_embed_proj_conv2_output.npy')
    output_tensor = torch.from_numpy(output_tensor)
    T,B,Cout,oh,ow = output_tensor.shape

    # 生成kenel，所有输出通道和输入通道的卷积核都使用0-8的顺序值
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)

    # convolution_result_inner_product = convolution_inner_product(input_tensor, kernel_tensor, padding=1, stride=1)
    convolution_result_out_product,total_add_num_out_product,total_cycle_num_out_product = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
    total_add_num_inner_product =  ow * oh * Cin * 9 * Cout * T * B
    # 比较结果是否相等
    # print("内积与外积结果是否一致：", torch.allclose(convolution_result_inner_product, convolution_result_out_product))
    print("外积所需累加次数：", total_add_num_out_product)
    print("内积所需累加次数：", total_add_num_inner_product)
    print("外积相较于内积所需累加次数：", total_add_num_out_product/total_add_num_inner_product)

    #估计latency
    #我的硬件参数
    pe_num = 4           #一个PE里计算oh=ow=4
    core_num= 8          #那就是一次可以计算8个Cout
    # total_cycle_num_out_product 是按照4 * 4串行处理得到的值，现在使用使用4个pe，8个core，会快32倍
    latency_estimation_out_product = (total_cycle_num_out_product/pe_num/core_num)
    # nvdla_large 的硬件参数
    Atomic_c = 64        #一次可以计算64个Cin
    nvdla_mac_num = 16   #那就是一次可以计算16个Cout
    # 稠密计算，
    latency_estimation_nvdla = oh * ow * 9  * T * B * math.ceil(Cout/nvdla_mac_num) * math.ceil(Cin/Atomic_c) 

    print("外积所需拍数：", latency_estimation_out_product)
    print("内积所需拍数：", latency_estimation_nvdla)
    print("加速比：", latency_estimation_nvdla/latency_estimation_out_product)



#测试模式2，作为硬件的golden，已经是padding后的值
    # T, Batch, Cin, in_h, in_w = 1, 1, 1, 6, 6
    # Cout = 1

    # # 生成全零输入张量
    # input_tensor = torch.zeros(T, Batch, Cin, in_h, in_w, dtype=torch.float32)

    # # i、j数组,由于tb中使用random，会出现重复的值
    # i= [2,1,3,5,3,0,3,1,1,3]
    # j= [3,5,5,4,1,5,4,0,0,3]
    # #将这些位置设置为1
    # for idx in range(len(i)):
    #     input_tensor[0,0,0,i[idx],j[idx]] = 1

    # print("输入张量：",input_tensor)

    # # 直接使用PyTorch创建卷积核张量：所有输出通道和输入通道的卷积核都使用0-8的顺序值
    # kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    # kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)

    # # 调用外积卷积函数
    # convolution_result_out_product, total_add_num_out_product = convolution_out_product(input_tensor, kernel_tensor, padding=0, stride=1)

    # print("软仿结果：")
    # print(convolution_result_out_product.squeeze())  # 去掉维度为1的维度并打印


#测试模式3，作为硬件的golden，已经padding后的值,pass
    # #10个输入通道，每个通道只有一个值为1
    # T, Batch, Cin, in_h, in_w = 1, 1, 10, 6, 6  # 10个输入通道
    # Cout = 1

    # # 生成全零输入张量
    # input_tensor = torch.zeros(T, Batch, Cin, in_h, in_w, dtype=torch.float32)

    # # 初始化初始值
    # i_val = 0
    # j_val = 0
    # k00, k01, k02 = 0, 1, 2
    # k10, k11, k12 = 3, 4, 5
    # k20, k21, k22 = 6, 7, 8

    # # 生成卷积核张量，用于测试
    # kernel_tensor = torch.zeros(Cout, Cin, 3, 3, dtype=torch.float32)

    # # 循环更新i、j和权重，共测试10拍
    # for cnt in range(10):
    #     # 更新i和j
    #     i_val = (i_val + 1) % 6  # i每次增加1，取模6
    #     j_val = (j_val + 2) % 6  # j每次增加2，取模6
        
    #     # 更新权重（递增）
    #     k00 = k00 + 1
    #     k01 = k01 + 1
    #     k02 = k02 + 1
    #     k10 = k10 + 1
    #     k11 = k11 + 1
    #     k12 = k12 + 1
    #     k20 = k20 + 1
    #     k21 = k21 + 1
    #     k22 = k22 + 1

    #     # 将权重设置到kernel_tensor中对应的位置
    #     kernel_tensor[0, cnt, :, :] = torch.tensor([
    #         [k00, k01, k02],
    #         [k10, k11, k12],
    #         [k20, k21, k22]
    #     ], dtype=torch.float32)

    #     # 在input_tensor的第cnt个通道的(i_val, j_val)位置设置为1
    #     input_tensor[0, 0, cnt, i_val, j_val] = 1

    #     # print(f"第{cnt}拍: i={i_val}, j={j_val}, 权重更新后: k00={k00_val}, k01={k01_val}, k02={k02_val}, "
    #     #       f"k10={k10_val}, k11={k11_val}, k12={k12_val}, k20={k20_val}, k21={k21_val}, k22={k22_val}")

    # print("\n输入张量非零元素位置:")
    # non_zero_indices = torch.nonzero(input_tensor)
    # for idx in non_zero_indices:
    #     print(f"通道{idx[2]}: 位置({idx[3]}, {idx[4]})")

    # # 调用外积卷积函数
    # convolution_result_out_product, total_add_num_out_product = convolution_out_product(input_tensor, kernel_tensor, padding=0, stride=1)

    # print("\n软仿结果：")
    # print(convolution_result_out_product.squeeze())  # 去掉维度为1的维度并打印
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    # 转换为分钟和秒
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60

    print(f"程序运行时间: {minutes}分{seconds:.2f}秒")

if __name__ == '__main__':
    main()