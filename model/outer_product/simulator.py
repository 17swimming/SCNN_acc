from outer_product.compress.shift import SplitUnit
# from core_copy import core
from utils import ceil_a_by_b, Stats
import torch
import os
import numpy as np
"""
只做性能模拟，不做真实计算

对于输入if[T,B,C,H,W],每一行经过shift单元后输出的bitstream长度就是cycle数
"""
class OutProductSimulator:
    """
    
    """
    def __init__(self, num_cores=16, num_pus=64):        
        self.num_cores = num_cores
        self.num_pus = num_pus
        self.GBsize_MP = 0
        self.GBsize_weight = 0
        self.bandwidth = 32
        self.weight_size = 16
        self.mp_size = 32
        self.lif_num = 32
        self.split_unit = SplitUnit()  # 创建SplitUnit实例

        # 数据搬运统计
        self.load_kernel_cycle = 0
        self.load_activation_cycle = 0
        self.read_mp_cycle = 0
        self.write_activation_cycle = 0
        self.all_adder_insufficient_positions = []

    def run_convolution(self, input_tensor, kernel_tensor, padding=1, stride=1):
        """
        input_tensor: [T, B, Cin, H, W] 或 [T, Cin, H, W] (B=1)
        kernel_tensor: [Cout, Cin, 3, 3]
        """
        self.global_stats = Stats()
        self.load_kernel_cycle = 0
        self.load_activation_cycle = 0
        self.compute_cycles = 0   # 总计算cycle
        self.total_cycles = 0     # 总cycle，考虑的数据搬运

        # 解析输入维度
        if len(input_tensor.shape) == 5:
            T, B, in_c, in_h, in_w = input_tensor.shape
        else:
            T, in_c, in_h, in_w = input_tensor.shape
            B = 1
            input_tensor = input_tensor.unsqueeze(1) 

        Cout, in_c_kernel, k_h, k_w = kernel_tensor.shape
        assert in_c == in_c_kernel, f"Cin mismatch: {in_c} vs {in_c_kernel}"

        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1
        self.GBsize_MP = self.num_pus * out_h * out_w * self.mp_size / 1024 / 8  # KB

        output_tensor = torch.zeros((T, B, Cout, out_h, out_w), dtype=kernel_tensor.dtype, device=input_tensor.device)

        # 一行pu对应一个cout
        Cout_group_nums = ceil_a_by_b(Cout, self.num_pus)

        for b in range(B):
            print(f"Processing batch {b}")
            # tag，一次需要把所有cin且num_pus个cout的权重加载到dram
            # todo，其实也可以在加载cin的act时，同时加载权重，但是为了简单，这里先不考虑
            kernel_data =  in_c * k_h * k_w * self.weight_size * self.num_pus
            self.GBsize_weight = kernel_data / 1024 / 8  # KB
            self.load_kernel_cycle = kernel_data / self.bandwidth
            self.global_stats.reads['dram'] += kernel_data
            self.global_stats.data_moved['kernel'] += kernel_data

            #tag 一组cout，算完所有时间步后再切换下一组，这样膜电位就无需搬运
            # for cout_group in range(Cout_group_nums):
            #tag 一个cout_group算完t0后，经过lif层得到spike，将spike写回dram，但是膜电位和权重都不offload至dram，加载下一个t的激活，
            for t in range(T):
                act_data = in_c * in_h * in_w
                self.load_activation_cycle = act_data / self.bandwidth
                self.global_stats.reads['dram'] += act_data
                self.global_stats.data_moved['act'] += act_data
                
                current_input = input_tensor[t, b, :, :, :]
                # 进行padding
                # 创建填充后的输入张量（注意：输入张量是[Cin,H,W]，填充只在H和W维度）
                padded_input = torch.zeros((in_c, in_h + 2 * padding, in_w + 2 * padding), 
                                        dtype=input_tensor.dtype, 
                                        device=input_tensor.device)
                padded_input[:, padding:padding+in_h, padding:padding+in_w] = current_input
                # 统计[C,H,W]中非0元素的个数
                ones_num_chw = padded_input.sum().item()
                
                # cin_cycle = []
                # for cin in range(in_c):
                #     cycle = 0
                #     current_input_tile = padded_input[cin, :, :]
                #     for r in range(current_input_tile.shape[0]):
                #         bitstream, (r, c_array) = self.split_unit.process(current_input_tile[r, :],r)
                #         cycle += len(bitstream)
                #     cin_cycle.append(cycle)
                    # 串行计算每个cin的cycle数
                    # self.compute_cycles += cycle  

                # 理想情况，直接调度至cycle最小的那个core
                # 切换t时需要进行同步，因此这个在这个时候复位core_cycles
                core_cycles = torch.zeros(self.num_cores, dtype=torch.int)
                for cin in range(in_c):
                #todo 对于每个core来说，只要切换了cin，就得重新配置一次core的权重
                    cycle = 0
                    current_input_tile = padded_input[cin, :, :]
                    for r in range(current_input_tile.shape[0]):
                        bitstream, (r, c_array) = self.split_unit.process(current_input_tile[r, :],r)
                        cycle += len(bitstream)
                    min_idx = torch.argmin(core_cycles)  
                    core_cycles[min_idx] += cycle  #tag，忽略配置core 权重的cycle
                # 统计算完所有cin的cycle数，并累加
                self.compute_cycles += torch.max(core_cycles).item()
                print(f"Batch {b}, Time {t}: Max core cycle: {torch.max(core_cycles).item()}; cin*h*w / self.num_cores: {in_c * in_h * in_w / self.num_cores}; ones_num_chw / self.num_cores: {ones_num_chw/self.num_cores}")
                print(f"--self.compute_cycles: {self.compute_cycles}")
                
                #所有cin都计算完成，则可以得到self.pus个cout的完整膜电位，下一层就是lif层
                lif_cycle_per_kernel = out_h * out_w  / self.lif_num
                lif_cycle = lif_cycle_per_kernel * self.num_pus

                # 把这组cout_group的lif结果写回dram
                self.global_stats.writes['dram'] += out_h * out_w * self.num_pus
                self.global_stats.data_moved['act'] += out_h * out_w * self.num_pus
                
                # # 保存cin_cycle数组到文件
                # output_dir = 'output/cin_cycles'
                # os.makedirs(output_dir, exist_ok=True)
                # filename = f'{output_dir}/cin_cycle_b{b}_t{t}.npy'
                # np.save(filename, cin_cycle)
                # print(f"Saved cin_cycle to {filename}")
                
                # # 计算并打印cin之间的负载均衡情况
                # if cin_cycle:
                #     mean_cycle = np.mean(cin_cycle)
                #     std_cycle = np.std(cin_cycle)
                #     max_cycle = max(cin_cycle)
                #     min_cycle = min(cin_cycle)
                #     print(f"Batch {b}, Time {t}: Cin cycle stats - Mean: {mean_cycle:.2f}, Std: {std_cycle:.2f}, Max: {max_cycle}, Min: {min_cycle}")
                
                # self.num_pus个cout为一组，上面仅统计了一组的cycle，再乘cout_group
                # tag，因为加载下一个t的激活时，lif层的输出也会写回dram，此处假设读和写act不能同时发生
                self.total_cycles += max(torch.max(core_cycles).item(), self.load_activation_cycle+lif_cycle)

        # 每个cout_group都是一样的，算完所有T，因此总cycle仅需要乘以cout_group_nums
        self.total_cycles = self.total_cycles * Cout_group_nums
        self.compute_cycles = self.compute_cycles * Cout_group_nums
        # 因为最后一个cout_group计算后，得等lif把num_pus个cout的膜电位都计算完成，所以额外需要lif_cycle,切换batch
        self.total_cycles += lif_cycle

        self.global_stats.total_cycles = self.total_cycles 
        self.global_stats.compute_cycles = self.compute_cycles 

        return output_tensor, self.global_stats

