from model.out_product import convolution_out_product
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def main():
    start_time = time.time()
    input_tensor = np.load('../conv/B=128/patch_embed_proj_conv3_input.npy') #TB合并了
    input_tensor = torch.from_numpy(input_tensor)
    print("input_tensor的稀疏度：", (input_tensor == 0).sum() / input_tensor.numel())
    
    # 拆分合并的TB维度
    TB, Cin, h, w = input_tensor.shape
    T = 4
    B = 128
    input_tensor = input_tensor.view(T, B, Cin, h, w)

    output_tensor = np.load('../conv/B=128/patch_embed_proj_conv3_output.npy')
    T,B,Cout,oh,ow = output_tensor.shape # 4, 128, 384, 16, 16

    # 生成kenel，所有输出通道和输入通道的卷积核都使用0-8的顺序值
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)

    # convolution_result_inner_product = convolution_inner_product(input_tensor, kernel_tensor, padding=1, stride=1)
    convolution_result_out_product,total_add_num_out_product,total_cycle_num_out_product,total_tile_num_out_product = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
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
    latency_estimation_nvdla = total_add_num_inner_product/(Atomic_c*nvdla_mac_num)

    print("外积所需拍数：", latency_estimation_out_product)
    print("内积所需拍数：", latency_estimation_nvdla)
    print("加速比：", latency_estimation_nvdla/latency_estimation_out_product)

    end_time = time.time()
    elapsed_seconds = end_time - start_time

    # 转换为分钟和秒
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60

    print(f"程序运行时间: {minutes}分{seconds:.2f}秒")

if __name__ == '__main__':
    main()