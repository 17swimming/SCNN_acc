import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math

# baseline
# 直接卷积结果，padding为1（用0填充），步长为1
def convolution_with_padding(input_matrix, kernel, padding=1, stride=1):
    # 获取输入矩阵和卷积核的尺寸
    in_h, in_w = input_matrix.shape
    k_h, k_w = kernel.shape
    
    # 计算输出矩阵的尺寸
    out_h = (in_h + 2*padding - k_h) // stride + 1
    out_w = (in_w + 2*padding - k_w) // stride + 1
    
    # 创建填充后的输入矩阵
    padded_input = torch.zeros((in_h + 2*padding, in_w + 2*padding), dtype=input_matrix.dtype, device=input_matrix.device)
    padded_input[padding:padding+in_h, padding:padding+in_w] = input_matrix
    
    # 初始化输出矩阵，使用与权重相同的数据类型
    output = torch.zeros((out_h, out_w), dtype=kernel.dtype, device=input_matrix.device)

    # 所需累加次数
    add_num = out_h * out_w * k_h * k_w
    
    # 执行卷积运算
    for i in range(out_h):
        for j in range(out_w):
            # 计算当前滑动窗口的位置
            h_start = i * stride
            h_end = h_start + k_h
            w_start = j * stride
            w_end = w_start + k_w
            
            # 提取窗口区域
            window = padded_input[h_start:h_end, w_start:w_end]
            
            # 执行卷积操作（元素相乘并求和）
            output[i, j] = torch.sum(window * kernel)
    
    return output, add_num

# 基于外积的卷积实现
def convolution_with_padding_out_product(input_matrix, kernel, padding=1, stride=1):
    # 获取输入矩阵和卷积核的尺寸
    in_h, in_w = input_matrix.shape
    k_h, k_w = kernel.shape
    
    # 计算输出矩阵的尺寸
    out_h = (in_h + 2*padding - k_h) // stride + 1
    out_w = (in_w + 2*padding - k_w) // stride + 1
    
    # 创建填充后的输入矩阵
    padded_input = torch.zeros((in_h + 2*padding, in_w + 2*padding), dtype=input_matrix.dtype, device=input_matrix.device)
    padded_input[padding:padding+in_h, padding:padding+in_w] = input_matrix
    # print("填充后的输入矩阵:") #没问题
    # print(padded_input)
    
    # 初始化输出矩阵，使用与输入相同的数据类型
    output = torch.zeros((out_h, out_w), dtype=kernel.dtype, device=input_matrix.device)

    add_num = 0
    tile_num = 0
    cycle_num = 0
    
    # eg：OF[0:3,0:3]对应的IF[0:5,0:5]，OF[0:3,4:7]对应的IF[0:5,4:9]————右边界是包含的
    # 将输出特征图按4×4分块处理，步长为4，即OF[a:a+4,b:b+4],对应的IF[a:a+6,b:b+6]————因为右边界是取不到的
    for a in range(0, out_h, 4):
        for b in range(0, out_w, 4):
            # 选取对应的IF
            IF = padded_input[a:a+6,b:b+6]
            # 使用我定义的卷积函数计算这一组的结果
            output[a:a+4,b:b+4], add_num_ij, cycle_num_ij = convolution_with_out_product(IF, kernel)
            add_num += add_num_ij
            cycle_num += cycle_num_ij
            tile_num += 1

    return output, add_num, cycle_num, tile_num
    
# 我定义的卷积函数，输入为6*6的稀疏矩阵和3*3的卷积核，输出为4*4的卷积结果
def convolution_with_out_product(input_matrix, kernel):
    # 获取输入矩阵和卷积核的尺寸,6*6 和 3*3
    in_h, in_w = input_matrix.shape
    k_h, k_w = kernel.shape

    # 计算输出矩阵的尺寸 4*4
    out_h = in_h  - k_h  + 1
    out_w = in_w  - k_w  + 1

    # 创建输出矩阵，使用与输入相同的数据类型
    output = torch.zeros((out_h, out_w), dtype=kernel.dtype, device=input_matrix.device)

    # 创建旋转180度后的卷积核
    kernel_180 = torch.rot90(kernel, k=2)
    # print("旋转180度后的卷积核:")
    # print(kernel_180)

    add_num = 0
    cycle_num = 0
    #对input_matrix每个'1'的横纵坐标(i,j)，进行如下操作：
    for i in range(in_h):
        for j in range(in_w):
            if input_matrix[i,j] == 1:
                # 第一步，取旋转180度后的卷积核的a-b行，c-d列
                # i,j在0-5之间，out_h,out_w为4
                a = 0 if(i > 2) else 2-i  
                # a = max(0,2-i)
                b = 2 if(i < out_h) else 5-i
                c = 0 if(j > 2) else 2-j  
                # c = max(0,2-j)
                d = 2 if(j < out_w) else 5-j
                kernel_add = kernel_180[a:b+1,c:d+1]

                # 第二步，确定kernel_add的放在输入矩阵的终点的位置(m,n)，以及起点位置(x,y)
                m = i if(i < out_h) else out_h-1
                n = j if(j < out_w) else out_w-1
                x = m - (b-a)
                y = n - (d-c)

                # 第三步，将kernel_add与输出矩阵中起点位置为(x,y)，kernel_add大小的这片区域做累加，即完成spike（i，j）的事件驱动计算
                output[x:m+1, y:n+1] += kernel_add

                # # 硬件测试模式2，打印i,j,ouput
                # print("----------------------i,j-------------------:\n",i,j)
                # print("output:\n",output)

                # 统计累加次数
                add_num += kernel_add.numel()
                # 假设每个spike仅需一拍即可完成计算
                cycle_num += 1

    return output,add_num,cycle_num            


# todo,定义一个函数，convolution_with_padding_out_product只能处理二维矩阵。这个函数实现Cin个通道的卷积。
# 怎么映射呢
#就拿不同排序算法来说，不同for循环写法会导致不同的计算量。
#这里有T维度，就得考虑怎么复用weight————似乎和计算量无关
#我的不同Cout可以并行————似乎也和计算量无关
#在统计计算量时，只要时for训练都当成在计算，所以是不是得考虑我们使用COO格式存储spike—和计算量有关
def convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1):   
    # 获取输入张量和卷积核的维度
    T, Batch, Cin, in_h, in_w = input_tensor.shape
    Cout, Cin, k_h, k_w = kernel_tensor.shape

    out_h = (in_h + 2*padding - k_h) // stride + 1
    out_w = (in_w + 2*padding - k_w) // stride + 1
    
    # 初始化输出张量
    output_tensor = torch.zeros(T, Batch, Cout, out_h, out_w, dtype=kernel_tensor.dtype, device=input_tensor.device)
    
    total_add_num = 0
    total_cycle_num = 0
    total_tile_num = 0
    # 移除重复的输出张量初始化
    # for T——B——Cin，统计每个in_h和in_w，最后结果再乘一个Cout就够了，因为都是相同的
    for t in range(T):
        for b in range(Batch):
                print(f"-----T: {t}, B: {b}------")
            # for cout in range(Cout):
                conv_result_cin = torch.zeros(output_tensor.shape[-2:], dtype=kernel_tensor.dtype, device=input_tensor.device)              
                for cin in range(Cin):
                    # 提取当前输入通道和卷积核
                    input_matrix = input_tensor[t, b, cin]
                    # kernel = kernel_tensor[cout, cin]
                    kernel = kernel_tensor[0, cin]  # 这里只处理一个通道来统计计算量和cycle
                                        
                    # 调用外积实现的卷积函数
                    conv_result_cin_np, add_num, cycle_num, tile_num = convolution_with_padding_out_product(input_matrix, kernel, padding, stride) 
                    
                    # 累加到输出中
                    conv_result_cin += conv_result_cin_np
                    total_add_num += add_num
                    total_cycle_num += cycle_num
                    total_tile_num += tile_num

                #    # 硬件测试模式3，打印通道，结果矩阵
                #     print(f"计算至输入通道{cin}，卷积核：\n",kernel)
                #     print(f"输出结果：\n",conv_result_cin)
                    print(f"计算至输入通道{cin}")
                # 将累加结果保存到输出张量中
                # print(f"计算至输出通道{cout}\n")
                # output_tensor[t, b, cout] += conv_result_cin
                output_tensor[t, b,0] += conv_result_cin
    
    # 注释了Cout，因为各个Cout的累加次数是一样的，直接乘就好了
    total_add_num = total_add_num * Cout
    total_cycle_num = total_cycle_num * Cout
    total_tile_num = total_tile_num * Cout
    return output_tensor, total_add_num, total_cycle_num, total_tile_num

# done,再定义一个函数，用于用内积完成所有通道的卷积
#先img2col，TBCHW——>M*K,  卷积核K*N ，输出M*N,使用矩阵乘完成计算，再reshape回TBCHW
def convolution_inner_product(input_tensor, kernel_tensor, padding=1, stride=1):
    # 获取输入张量和卷积核的维度
    T, Batch, Cin, in_h, in_w = input_tensor.shape
    Cout, Cin, k_h, k_w = kernel_tensor.shape
    
    # -------------------------- 步骤1：展平T和B维度，[T,B,Cin,in_h,W] -> [TB, Cin, in_h, W] --------------------------
    TB = T * Batch
    input_flattened = input_tensor.reshape(TB, Cin, in_h, in_w)
    
    # -------------------------- 步骤2：对输入进行padding操作（保持张量维度顺序，填充H和W维度） --------------------------
    # padding=(0,0,padding,padding) 对应 (左,右,上,下)，仅对H（高度）和W（宽度）维度填充
    input_padded = torch.nn.functional.pad(input_flattened, (padding, padding, padding, padding), mode='constant', value=0)
    
    # -------------------------- 步骤3：计算输出特征图的尺寸（H_out, W_out） --------------------------
    # 卷积输出尺寸公式：H_out = [(H_in + 2*padding - k_h) // stride] + 1
    out_h = (in_h + 2 * padding - k_h) // stride + 1
    out_w = (in_w + 2 * padding - k_w) // stride + 1
    
    # -------------------------- 步骤4：img2col转换，[TB, Cin, H_pad, W_pad] -> [M, K] --------------------------
    # 其中 M = TB * out_h * out_w（所有输出像素的总数）
    # K = Cin * k_h * k_w（每个卷积窗口的元素总数，即卷积核展平后的维度）
    
    # 创建一个空的矩阵来存储所有的卷积窗口
    M = TB * out_h * out_w
    K = Cin * k_h * k_w
    input_col = torch.zeros(M, K)
    
    # 手动提取所有卷积窗口
    # 遍历每个输出位置
    row_idx = 0
    for tb in range(TB):
        for h_out in range(out_h):
            for w_out in range(out_w):
                # 提取对应位置的k_h x k_w卷积窗口
                h_start = h_out * stride
                h_end = h_start + k_h
                w_start = w_out * stride
                w_end = w_start + k_w
                
                # 提取k_h x k_w区域（考虑padding后）
                patch = input_padded[tb, :, h_start:h_end, w_start:w_end]
                
                # 将patch展平为向量并存储
                input_col[row_idx, :] = patch.flatten()  # Shape: [Cin*k_h*k_w]
                row_idx += 1
    
    # print("input_col :", input_col)
    # -------------------------- 步骤5：卷积核展平，[Cout, Cin, k_h, k_w] -> [K, Cout]（对应题目中的K*N，N=Cout） --------------------------
    kernel_col = kernel_tensor.reshape(Cout, K).t()  # 转置后维度为 [K, Cout]，匹配矩阵乘法维度要求
    
    # -------------------------- 步骤6：矩阵乘法实现卷积计算，[M, K] @ [K, Cout] -> [M, Cout] --------------------------
    output_mat = torch.matmul(input_col, kernel_col)
    
    # -------------------------- 步骤7：重塑结果回 [T, B, Cout, out_h, out_w] --------------------------
    # 步骤7.1：将output_mat重塑为 [TB, out_h, out_w, Cout]
    output_reshaped = output_mat.reshape(TB, out_h, out_w, Cout)
    
    # 步骤7.2：转置为 [TB, Cout, out_h, out_w]
    output_transposed = output_reshaped.permute(0, 3, 1, 2)
    
    # 步骤7.3：将TB维度拆分为 [T, B]
    output_final = output_transposed.reshape(T, Batch, Cout, out_h, out_w)
    
    return output_final

def main():
    start_time = time.time()
#使用真实数据测试
    #读取npy文件,lif3的输出不是rpe卷积的输入，maxpool3的输出才是rep卷积的输入
    # input_tensor = np.load('../conv/patch_embed_proj_lif3_output.npy')
    input_tensor = np.load('../conv/patch_embed_rpe_conv_input.npy') #TB合并了
    input_tensor = torch.from_numpy(input_tensor)
    print("input_tensor的稀疏度：", (input_tensor == 0).sum() / input_tensor.numel())
    
    # 拆分合并的TB维度
    TB, Cin, h, w = input_tensor.shape
    T = 4
    B = 128
    input_tensor = input_tensor.view(T, B, Cin, h, w)

    T,B,Cout,oh,ow = 4, 128, 384, 8, 8

    # 生成kenel，所有输出通道和输入通道的卷积核都使用0-8的顺序值
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)

    # convolution_result_inner_product = convolution_inner_product(input_tensor, kernel_tensor, padding=1, stride=1)
    total_add_num_inner_product =  ow * oh * Cin * 9 * Cout * T * B
    convolution_result_out_product,total_add_num_out_product,total_cycle_num_out_product,total_tile_num_out_product = convolution_out_product(input_tensor, kernel_tensor, padding=1, stride=1)
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