
from utils import img2col
import torch
import numpy as np
from collections import OrderedDict
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Stats, ceil_a_by_b

#  输入3*3的bitmap，返回累加结果和cycle
class PE:
    """
    PE中weights仅在切换cout和cin时进行切换
    其余时刻pe就是接收3*3 bits的act，然后执行计算，输出cycle和计算结果就好
    """
    #每次调用 pe.process() 时， add_num 和 cycle_num 只反映当前 act 的统计信息
    #pe.stats 则反映该 PE 自创建以来处理所有 act 的累计统计信息
    def __init__(self, weights, adder_num, name, pe_id):
        self.pe_id = pe_id
        self.name = name
        self.stats = Stats()   
        self.adder_num = adder_num
        self.cycle_num = 0
        self.weights = weights #9个权重

    def process(self, act):
        """
        处理3*3的act（0/1值），返回累加结果psum和计算cycle
        
        Args:
            act: 3*3的tensor，值为0或1，1表示对应的weight需要被累加
            
        Returns:
            psum: 累加结果
            cycle: 计算所需的cycle数
        """
        # 获取act中1的索引（非零元素的位置）
        nonzero_indices = torch.nonzero(act, as_tuple=False)
        # 将二维坐标转换为一维索引（0-8）
        one_d_indices = nonzero_indices[:, 0] * 3 + nonzero_indices[:, 1]
        # 取对应的权重
        w_valid = self.weights[one_d_indices]
        one_num = len(nonzero_indices)
        psum = 0
        
        # 如果没有非零元素，直接返回0和0 cycle
        if one_num == 0:
            return psum, 0
        
        # 初始化当前act的操作数
        self.add_num = 0
        self.cycle_num = 0
        
        # 每拍可以处理adder_num+1个weights
        weights_per_cycle = self.adder_num + 1
        
        # 计算需要的cycle数
        self.cycle_num = ceil_a_by_b(one_num, weights_per_cycle)
        
        # 逐cycle累加
        for cycle in range(self.cycle_num):
            # 计算当前cycle处理的weights范围
            start_idx = cycle * weights_per_cycle
            end_idx = min(start_idx + weights_per_cycle, one_num)
            
            # 当前cycle要处理的weights
            current_weights = w_valid[start_idx:end_idx]
            
            # 累加到psum
            for w in current_weights:
                psum += w
                self.add_num += 1
        
        # 更新统计信息，持续统计
        self.stats.num_ops += self.add_num
        self.stats.compute_cycles += self.cycle_num  
        
        return psum, self.cycle_num 
        


class Core:
    """
    核心计算单元，根据加速器配置实现功能
    每个core对应一个Cout，包含4个PE，PE之间并行工作
    """
    def __init__(self, name, cout_idx, num_pes=64, adder_num=2):
        self.name = name
        self.cout_idx = cout_idx  # 对应哪个Cout
        self.num_pes = num_pes
        # PE初始化：weights, adder_num, name, pe_id
        # 这里暂时用默认值，实际使用时需要设置weights
        # 初始化PE列表
        self.pes = [PE(torch.zeros(9), adder_num, f"pe{i}", i) for i in range(num_pes)]
        self.stats = Stats()    # 这个有啥用，为啥要重复统计
        self.total_cycle_num = 0     # core中4级流水线的总周期数
        self.compute_cycles_num = 0  # 计算周期数
        self.print_info = True  # 控制是否打印core信息
        
    def configure_core(self, kernel=None):
        """
        配置加速器参数
        
        Args:
            kernel: 卷积核张量，形状为 [processed_cin, 3, 3]，用于为每个PE配置weights
        """
        # 如果提供了kernel，为每个PE配置对应的weights
        if kernel is not None:
            # 确保kernel形状正确，但是会出现cin不够的情况
            # assert kernel.shape[0] == self.num_pes, f"Kernel Cin ({kernel.shape[0]}) must match num_pes ({self.num_pes})"
            # assert kernel.shape[1] == 3 and kernel.shape[2] == 3, f"Kernel must be 3x3, got {kernel.shape[1:3]}"
            
            # 为每个PE分配对应的权重
            # 如果kernel的通道数小于PE数量，只配置前kernel.shape[0]个PE
            num_valid_channels = kernel.shape[0]
            for i, pe in enumerate(self.pes):
                if i < num_valid_channels:
                    # 将3x3的kernel展平为9个元素的一维张量
                    pe.weights = kernel[i, :, :].reshape(-1)
                else:
                    # 超出kernel通道数的PE配置为零权重
                    pe.weights = torch.zeros(9, dtype=kernel.dtype, device=kernel.device)
        
    # padding 在core内完成
    def process(self, input_tensor, kernel_tensor, padding=1, stride=1):
        """
        处理整个输入张量和对应的卷积核张量，输出1xHxW的结果（对应一个Cout）
        
        Args:
            input_tensor: 64*HxW的输入张量
            kernel_tensor: 64*3x3的卷积核张量（对应一个Cout）
            padding: 填充大小
            stride: 步长
            
        Returns:
            output_tensor: 1xHxW的输出张量
            cycle: 计算周期数
        """
        # 获取输入和卷积核的尺寸
        in_c, in_h, in_w = input_tensor.shape
        in_c_kernel, k_h, k_w = kernel_tensor.shape
        
        # 确保Cin一致
        assert in_c == in_c_kernel, f"Cin mismatch: input={in_c}, kernel={in_c_kernel}"
        
        # 计算输出张量的尺寸
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1
        
        # 创建填充后的输入张量（注意：输入张量是[Cin,H,W]，填充只在H和W维度）
        padded_input = torch.zeros((in_c, in_h + 2 * padding, in_w + 2 * padding), 
                                   dtype=input_tensor.dtype, 
                                   device=input_tensor.device)
        padded_input[:, padding:padding+in_h, padding:padding+in_w] = input_tensor
        
        # 初始化输出张量 (1xHxW)
        output_tensor = torch.zeros((1, out_h, out_w), dtype=kernel_tensor.dtype, device=input_tensor.device)
        
        # 重置统计信息
        self.compute_cycles_num= 0
        self.total_cycle_num = 0
        self.stats = Stats()

        # 也重置所有 PE 的统计信息
        for pe in self.pes:
            pe.stats = Stats()
        
        # 权重数据固定到PE中
        # kernel_tile = kernel_tensor[:, :, :, :]

        # 流水线，取数-PE计算-加法树累加
        # 遍历输出张量的每个位置
        for h in range(out_h):
            for w in range(out_w):
                #----------------取数逻辑----------------
                # 计算当前tile的起始位置
                h_start, w_start = h * stride, w * stride
                h_end, w_end = h_start + k_h, w_start + k_w
                input_tile = padded_input[:, h_start:h_end, w_start:w_end]
                # ----------------PE计算逻辑----------------
                #  PE输入3*3的bitmap，返回累加结果和cycle
                psum = [0] * self.num_pes
                cycle = [0] * self.num_pes
                num_valid_channels = input_tile.shape[0]
                for i, pe in enumerate(self.pes):
                    # 每个PE处理一个输入通道的3x3 tile
                    # 如果i超出实际输入通道数，使用零输入（对应零权重PE）
                    if i < num_valid_channels:
                        pe_input = input_tile[i, :, :]
                    else:
                        pe_input = torch.zeros((k_h, k_w), dtype=input_tile.dtype, device=input_tile.device)
                    psum[i], cycle[i] = pe.process(pe_input)
                # 选取所有pe中cycle数最大的那个作为该input_tile的计算cycle
                self.compute_cycles_num += max(cycle)
                # if(max(cycle) > 1):
                #     print(f'加法器数量为:{self.pes[0].adder_num}')
                #     print(f'当前64*3*3的输入中1的数目为:{input_tile.sum()}')
                #     print(f'计算当前64*3*3输入的cycle为:{max(cycle)}')
                
                # # ----------------加法树累加逻辑----------------
                # # 一拍完成累加
                # # 64个psum，4输入加法器，第一级累加，得到16个psum
                # for i in range(0, self.num_pes, 4):
                #     psum[i//4] = psum[i] + psum[i+1] + psum[i+2] + psum[i+3]
                # # 第二级累加得到4个psum，每个psum对应一个Cout
                # for i in range(0, self.num_pes//4, 2):
                #     psum[i//2] = psum[i] + psum[i+1]
                # # 最后一级得到最终结果
                # output_tensor[0, h, w] = psum[0] 

                # # MP写回逻辑
        
        # 每个core计算时，我们认为active和权重都已经在片上RAM中了，所以不加RAM时间 
        # 总cycle =  pe的计算cycle + 1拍取数 + 加法树 + 写MP
        print(f'理想情况1拍一个post_nu，实际{self.compute_cycles_num / (out_h * out_w)}')
        self.total_cycle_num = self.compute_cycles_num + 1 + 2

        self.stats.compute_cycles += self.compute_cycles_num
        return output_tensor, self.total_cycle_num
        
  

class InnerProductSimulator:
    """
    外积卷积模拟器，根据加速器配置实现功能
    包含8个core，每个core对应一个Cout，所有core共享相同的input_tensor
    """
    def __init__(self):
        self.num_cores = 64  # bank的深度
        self.num_pes = 16  # 每个pe负责计算K中的一小段，比如9*16，那么就是16个cin
        self.RAM_size = 32*32*64
        self.bandwidth = 32
        self.weight_size = 16          # 16 btis的权重
        self.mp_size = 32              # 32 btis的MP
        self.adder_num = 2              #每个pe中加法器数量   
        self.load_kernel_cycle = 0     # 切换kernel 所需 cycle
        self.load_activation_cycle = 0  # 切换activation 所需 cycle
        self.read_mp_cycle = 0  # 读MP 所需 cycle
        self.write_activation_cycle = 0  # 写activation 所需 cycle
        self.compute_cycles = 0           # batch-cin-h-w四重循环计算完的计算cycle
        self.total_cycles = 0
        self.lif_num     = 1

        self.processed_cin = self.num_pes  #每个pe对应一个cin
        self.cores = [Core(f"core{i}", i, self.num_pes, self.adder_num) for i in range(self.num_cores)]
        self.global_stats = Stats()
        self.print_core_info = True  # 控制是否打印core信息，只打印一次
        
    def configure_accelerator(self, kernel=None):
        """
        配置所有核心的加速器参数
        
        Args:
            kernel: 卷积核张量，形状为 [Cout, processed_cin, 3, 3]，用于为每个core的PE配置weights
        """
        for i, core in enumerate(self.cores):
            # 如果提供了kernel，为每个core分配对应的Cout权重
            core_kernel = kernel[i, :, :, :] if kernel is not None else None
            core.configure_core(core_kernel)
    
    def run_convolution(self, input_tensor, kernel_tensor, padding=1, stride=1):
        """
        Args:
            input_tensor: 5D输入张量 [T,B,Cin,H,W]
            kernel_tensor: 4D权重张量 [Cout,Cin,k_h,k_w]
            padding: 填充大小
            stride: 步长
            
        Returns:
            output_tensor: 5D输出张量 [T,B,Cout,H_out,W_out]
            stats: 全局统计信息
        """                
        # 重置全局统计信息
        self.global_stats = Stats()
        self.compute_cycles = 0
        self.total_cycles = 0
        
        # 判断输入shape
        if len(input_tensor.shape) == 5:
            T, B, in_c, in_h, in_w = input_tensor.shape
        elif len(input_tensor.shape) == 4:
            # batch=1，4D tensor的shape是 [T, in_c, in_h, in_w]
            B = 1
            T, in_c, in_h, in_w = input_tensor.shape

        Cout, in_c_kernel, k_h, k_w = kernel_tensor.shape

        # 确保Cin一致
        assert in_c == in_c_kernel, f"Cin mismatch: input={in_c}, kernel={in_c_kernel}"
        
        # 计算输出尺寸
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1
        
        # 初始化输出张量 (T,B,Cout,H_out,W_out)
        output_tensor = torch.zeros((T, B, Cout, out_h, out_w), dtype=kernel_tensor.dtype, device=input_tensor.device)

        #先通过img2col得到权重矩阵[K,N]和激活矩阵[B,T,M,K]
        kernel_tensor = kernel_tensor.view(Cout, -1)  # [Cout, K]
        for b in range(B):
            input_tensor[b] = img2col(input_tensor[:,b,:,:,:], k_h, padding, stride)  # [T, M, K]

                
        # ------------------- 切换权重------------------
        # 从DRAM加载self.num_cores个核的processed_cin个输入通道至core中，即每个core对应一个cout，然后core中每个PE放kernel的一个cin
        # 加载的self.nm_cores个核的processed_cin个输入通道,只有在处理完所有t、b后，才需要加载剩余的cin直至一个kernel的所有cin计算完成
        # 算完一个group，再加载下一个group的核至core中
        # 对Cout进行拆分，num_cores个Cout为一个group，每个group一起计算，group内每个Cout对应一个core
        Cout_group = ceil_a_by_b(Cout,self.num_cores)
        # print(f"{Cout}个输出通道,需要分成{Cout_group}个group")
        # tag，如果精确计算，这里还需要一个for 来切换group，但是由于每个group的计算cycle一致，所以直接乘就好
        # for cout_group_idx in range(Cout_group):

        #tag,每个核处理时，先把所有时间步算完，膜电位就没有片上和片外的搬运，因为batch之间膜电位独立，time step之间膜电位有顺序
        for b in range(B):
            print(f"---------开始处理batch{b}---------")
            # --------------------batch之间需要切换MP------------------
            # 由于我把time_steps放在了内层循环，每个batch开始计算时无需搬运mp
            # mp_data = self.num_cores * out_h * out_w * self.mp_size
            # self.read_mp_cycle += mp_data / self.bandwidth
            # self.global_stats.reads['dram'] += mp_data
            # self.global_stats.data_moved['mp'] += mp_data

            # ------------------- 切换输入通道------------------
            # 我一次只能处理self.processed_cin个Cin,先把Cin拆分成多个channel_operation，每个channel_operation处理self.processed_cin个Cin,
            channel_operation = ceil_a_by_b(in_c,self.processed_cin)  # 向上取整
            # 同一个cout，所有时间步下，输出神经元的膜电位计算完成
            cin_T_h_w_computed_cycles = 0   
            cin_T_h_w_total_cycles = 0
            for cin_group_idx in range(channel_operation):
                # 计算当前cin group的Cin起始和结束索引
                cin_start = cin_group_idx * self.processed_cin
                cin_end = min(cin_start + self.processed_cin, in_c)

                # ---------先从DRAM加载权重，假设只能串行加载每个core的kernel
                kernel_data = self.num_cores * (cin_end - cin_start) * k_h * k_w * self.weight_size
                self.load_kernel_cycle += kernel_data / self.bandwidth
                self.global_stats.reads['dram'] += kernel_data
                self.global_stats.data_moved['kernel'] += kernel_data
                
                # 获取当前group的Cout对应的卷积核 [num_cores, processed_cin, k_h, k_w]
                # 假设每个group处理num_cores个Cout
                current_kernel = kernel_tensor[0, cin_start:cin_end, :, :]
                
                # 取权重后配置core，直至算完所有timesteps，mem_potential就无需再切换
                core = self.cores[0]  
                core.configure_core(current_kernel)
                
                # 统计一个一组cin计算所需的周期
                T_h_w_computed_cycles = 0
                T_h_w_total_cycles = 0
                for t in range(T):
                    # tag,没有精确计算，仅计算group0，用于性能评估
                    # print("-batch:", b, "-time step:", t,"-")
                    # ---------从DRAM加载激活-----------------
                    # todo，获取当前T和B的输入张量 [processed_cin,H,W]，            
                    current_input = input_tensor[t, b, cin_start:cin_end, :, :]
                    act_data = (cin_end - cin_start) * in_h * in_w 
                    self.load_activation_cycle = act_data / self.bandwidth
                    self.global_stats.reads['dram'] += act_data
                    self.global_stats.data_moved['act'] += act_data

                    # print(f'加法器数量为:{core.pes[0].adder_num}') 
                    # 执行卷积计算（所有core共享相同的input_tensor），这个时候的current_input还没有padding
                    if self.print_core_info:
                        
                        output_tile, core_compute_cycles= core.process(current_input, current_kernel, padding, stride)
                        self.print_core_info = False  # 打印一次后关闭
                    else:
                        output_tile, core_compute_cycles = core.process(current_input, current_kernel, padding, stride)
                    
                    # tag，将当前64*3*3的结果写入输出张量，这里需要累加
                    output_tensor[t, b, 0, :, :] += output_tile[0, :, :]
                    
                    # 汇总统计信息（core并行工作，周期数都一样）
                    # core.process时就会刷新统计信息，所以需要一直累加
                    T_h_w_computed_cycles += core_compute_cycles   
                    T_h_w_total_cycles += core_compute_cycles + self.load_activation_cycle
                    # print(f"T:{t}，cin:{cin_start}-{cin_end}时，计算周期数：{core_compute_cycles}，总周期数：{core_compute_cycles + self.load_activation_cycle}")
                cin_T_h_w_computed_cycles += T_h_w_computed_cycles
                cin_T_h_w_total_cycles += T_h_w_total_cycles
                # print(f"cin_group{cin_group_idx}的计算周期数：{T_h_w_computed_cycles}，总周期数：{T_h_w_total_cycles}")
            # todo,后接lif层，因为所有cin和timestep都算完了，mp已经完整了，所以直接接lif层
            lif_cycle = out_h * out_w * T / self.lif_num

            self.compute_cycles += cin_T_h_w_computed_cycles
            # 膜电位累计和lif层激活是并行的，所以取最大的那个周期数
            self.total_cycles += max(cin_T_h_w_total_cycles, lif_cycle)
        # ------------------- 把spike写回DRAM ------------------
        activation_write_data = Cout * out_h * out_w 
        self.write_activation_cycle = activation_write_data / self.bandwidth
        self.global_stats.writes['dram'] += activation_write_data
        self.global_stats.data_moved['act'] += activation_write_data
        print(f"一个输出通道的计算所需计算周期为:{T_h_w_computed_cycles},总周期为:{T_h_w_total_cycles}")
        
        
        #由于每个group的计算过程相同，周期数只需要计算一次然后乘以循环次数
        self.global_stats.total_cycles = self.total_cycles * Cout_group
        self.global_stats.compute_cycles = self.compute_cycles * Cout_group
            
        return output_tensor, self.global_stats

# 示例用法
if __name__ == "__main__":    

# 测试5D输入和4D权重
    print("\n=== 5D输入和4D权重测试 ===")
    T, B, Cin, H, W = 1, 1, 8, 8, 4  #会有4个stride操作，每个pe复用4个tile，所以总共有16个tile被处理
    Cout = 32
    
    # 创建5D输入张量 [T,B,Cin,H,W]
    dense = 0.1
    input_tensor = torch.zeros(T, B, Cin, H, W)
    torch.manual_seed(42)
    num_ones = int(T * B * Cin * H * W * dense)  # 10%的稠密度
    indices = torch.randperm(T*B*Cin*H*W)[:num_ones]
    input_tensor.view(-1)[indices] = 1
    
    # 创建4D权重张量 [Cout,Cin,k_h,k_w]
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)
    
    simulator = InnerProductSimulator()
    simulator.num_pes = 8
    simulator.num_cores = 16
    simulator.adder_num = 2
    simulator.print_core_info = True 
    padding = 1
    stride = 1
    output_tensor, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=padding, stride=stride)

    oh =( H - kernel_tensor.shape[2] + 2 * padding) // stride + 1
    ow = ( W - kernel_tensor.shape[3] + 2 * padding) // stride + 1

    # 我理论估计我需要的计算周期数,理论一拍一个输出神经元
    theorical_cycles = oh * ow * T * B * math.ceil(Cout/16) * math.ceil(Cin/64) 
    # 稠密计算，
    latency_estimation_nvdla = H * W * 9  * T * B * math.ceil(Cout/16) * math.ceil(Cin/64) 

    
    print(f"模拟器输入形状: {input_tensor.shape}")
    print(f"模拟器输出形状: {output_tensor.shape}")
    print(f"模拟器总周期: {stats.total_cycles}")
    print(f"模拟器计算周期: {stats.compute_cycles}")
    print(f"理论周期: {theorical_cycles}")
    print(f"dla的计算周期: {latency_estimation_nvdla}")
    print(f"计算加速比：{latency_estimation_nvdla / stats.compute_cycles:.2f}")
    
    # 打印数据量统计
    print("\n=== 数据量统计 ===")
    print(f"从DRAM搬运的MP数据量: {stats.data_moved['mp']} bits")
    # 从DRAM搬运的kernel数据量=b * cout * cin * k_h * k_w * weight_size,因为每个batch都需要搬运一次
    print(f"从DRAM搬运的kernel数据量: {stats.data_moved['kernel']} bits")
    print(f"从DRAM搬运的act数据量: {stats.data_moved['act']} bits")
    total_data = stats.data_moved['mp'] + stats.data_moved['kernel'] + stats.data_moved['act']
    print(f"总数据量: {total_data} bits ({total_data / 8 / 1024:.2f} KB)")
