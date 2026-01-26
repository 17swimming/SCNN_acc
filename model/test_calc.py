from rpe_conv import convolution_out_product
import torch
import numpy as np
from out_product_simulator import OutProductSimulator
import math
import time

def test_calc():
    """
    测试计算是否正确
    """
    start_time = time.time()
    print("=== 测试proj conv1 计算是否正确 ===")
    input_tensor = np.load('../conv/B=2/patch_embed_proj_conv1_input.npy') #TB合并了
    input_tensor = torch.from_numpy(input_tensor)
    TB, Cin, h, w = input_tensor.shape
    T = 4
    B = 2
    input_tensor = input_tensor.view(T, B, Cin, h, w)

    output_tensor_golden = np.load('../conv/B=2/patch_embed_proj_conv1_output.npy')
    output_tensor_golden = torch.from_numpy(output_tensor_golden)
    TB, Cout, oh, ow = output_tensor_golden.shape
    T = 4
    B = 2
    output_tensor_golden = output_tensor_golden.view(T, B, Cout, oh, ow)

    # 从检查点文件中读取rpe层的权重参数，文件地址为\spikformer\model_best.pth.tar
    # 这里直接使用已经提取出来的权重npy文件就好
    kernel_tensor = torch.from_numpy(np.load('../conv/B=2/patch_embed_proj_conv1_weight.npy'))

    # 使用模拟器完整计算
    simulator = OutProductSimulator()
    output_tensor_sim, global_stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)
    end_time_sim = time.time()
    # 使用out_product函数估计
    convolution_result_out_product,total_add_num_out_product,total_cycle_num_out_product,total_tile_num_out_product = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
    
    print(f"模拟器所需时间：{end_time_sim - start_time}")
    if torch.allclose(output_tensor_sim.float(), output_tensor_golden.float()):
        print("✅ 模拟器计算结果与golden输出一致")
    else:
        print("❌ 模拟器计算结果与golden输出不一致")
    # convolution_out_product函数仅算了一个Cout，所以仅比较Cout = 0的输出
    if torch.allclose(convolution_result_out_product[:, :, 0, :, :].float(), output_tensor_golden[:, :, 0, :, : :].float()):
        print("✅ convolution_out_product计算结果与golden输出一致")
    else:
        print("❌ convolution_out_product计算结果与golden输出不一致")
    print(f"总操作数: {total_add_num_out_product}")

    #估计latency
    # nvdla_large 的硬件参数
    Atomic_c = 64        #一次可以计算64个Cin
    nvdla_mac_num = 16   #那就是一次可以计算16个Cout
    latency_estimation_nvdla = oh * ow * 9  * T * B * math.ceil(Cout/nvdla_mac_num) * math.ceil(Cin/Atomic_c) 

    print("convolution_out_product所需拍数：", total_cycle_num_out_product)
    print("模拟器所需拍数：", global_stats.total_cycles)
    print("内积所需拍数：", latency_estimation_nvdla)
    print("convolution_out_product加速比：", latency_estimation_nvdla/total_cycle_num_out_product)
    print("模拟器加速比：", latency_estimation_nvdla/global_stats.total_cycles)

if __name__ == "__main__":
    test_calc()
