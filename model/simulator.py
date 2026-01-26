import torch
import numpy as np
from collections import OrderedDict
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import Stats, ceil_a_by_b

class PE:
    """
    处理元素(PE)，按照加速器配置实现功能
    每个PE处理一块4x4的输出特征图，每次只能计算一个Cin，因此需要串行处理整个块
    """
    #每次调用 pe.process() 时， add_num 和 cycle_num 只反映当前 tile 的统计信息
    #pe.stats 则反映该 PE 自创建以来处理所有 tile 的累计统计信息
    def __init__(self, name, pe_id):
        self.name = name
        self.pe_id = pe_id
        self.stats = Stats()   
        self.add_num = 0
        self.cycle_num = 0
    def process(self, input_block, kernel_block):
        """
        处理6x6xCin输入块和3x3xCin卷积核块，输出1x4x4的结果
        
        Args:
            input_block: 6x6xCin的输入张量块
            kernel_block: 3x3xCin的卷积核张量块
            
        Returns:
            output_block: 1x4x4的输出张量块
            add_num: 累加操作次数
            compute_cycles: 计算周期数
            sparsity: 输入块的稀疏度
        """
        # 获取输入和卷积核的尺寸
        in_c, in_h, in_w = input_block.shape
        in_c_kernel, k_h, k_w = kernel_block.shape
        
        # 确保Cin一致
        assert in_c == in_c_kernel, f"Cin mismatch: input={in_c}, kernel={in_c_kernel}"
        
        # 计算输出块的尺寸
        out_h = in_h - k_h + 1
        out_w = in_w - k_w + 1
        
        # 初始化输出块 (1x4x4)
        output_block = torch.zeros((1, out_h, out_w), dtype=kernel_block.dtype, device=input_block.device)
        
        # 每次都会重置，便于统计每个tile的统计信息
        self.add_num = 0
        self.cycle_num = 0
        
        # todo，在pe内创建旋转180度后的卷积核（仅对H和W维度旋转）
        kernel_180 = torch.rot90(kernel_block, k=2, dims=[1, 2])
        
        # # 串行处理每个Cin（根据配置，每次只能计算一个Cin）
        # for cin in range(in_c):
        #     # 提取当前Cin的输入和卷积核
        #     current_input = input_block[cin, :, :]
        #     current_kernel = kernel_180[cin, :, :]
            
        #     # 对输入矩阵每个非零元素进行事件驱动计算
        #     for i in range(in_h):
        #         for j in range(in_w):
        #             if current_input[i, j] != 0:  # 支持非二进制输入
        #                 # 确定卷积核旋转后的有效区域
        #                 a = max(0, 2 - i)
        #                 b = 2 if (i < out_h) else 5 - i
        #                 c = max(0, 2 - j)
        #                 d = 2 if (j < out_w) else 5 - j
        #                 kernel_add = current_kernel[a:b+1, c:d+1]
                        
        #                 # 确定有效区域在输出矩阵中的位置
        #                 m = i if (i < out_h) else out_h - 1
        #                 n = j if (j < out_w) else out_w - 1
        #                 x = m - (b - a)
        #                 y = n - (d - c)
                        
        #                 # 累加操作
        #                 output_block[0, x:m+1, y:n+1] += current_input[i, j] * kernel_add
                        
        #                 # 统计累加次数
        #                 self.add_num += kernel_add.numel()
        #                 # 假设每个spike仅需一拍即可完成计算
        #                 self.cycle_num += 1
        
        # 一次处理k个cin，对k*in_h*in_w大小的块，将脉冲的位置拆分至3个数组
        # nornal列表存储j < oh and i < ow的，
        # sub列表存储j >= oh 的
        # right列表存储j < oh and i >= ow的，
        # 三个列表中数据每次弹出一个，这样称为一个pack。每个pack中根据三个i、j分别去取3个kernel_add以及xymn，然后再和output_block做累加，即一次处理三个脉冲了
        # 注意，需要区分Cin，因为不同cin的kernel_add是不同的，所以需要根据cin来取不同的kernel_add
        k = 1  # 这里k=1，根据实际配置可以调整
        for cin_group in range(0, in_c, k):
            # 提取当前组的Cin范围
            cin_end = min(cin_group + k, in_c)
            current_input_group = input_block[cin_group:cin_end, :, :]
            current_kernel_group = kernel_180[cin_group:cin_end, :, :]
            
            # 初始化三个列表
            normal_list = []  # 存储j < oh and i < ow的
            sub_list = []     # 存储j > oh 的
            right_list = []   # 存储j < oh and i > ow的
            
            # 收集所有非零元素的位置和值
            for cin in range(current_input_group.shape[0]):
                for i in range(in_h):
                    for j in range(in_w):
                        if current_input_group[cin, i, j] != 0:  # 支持非二进制输入
                            # 确定卷积核旋转后的有效区域
                            a = max(0, 2 - i)
                            b = 2 if (i < out_h) else 5 - i
                            c = max(0, 2 - j)
                            d = 2 if (j < out_w) else 5 - j
                            kernel_add = current_kernel_group[cin, a:b+1, c:d+1]
                            
                            # 确定有效区域在输出矩阵中的位置
                            m = i if (i < out_h) else out_h - 1
                            n = j if (j < out_w) else out_w - 1
                            x = m - (b - a)
                            y = n - (d - c)
                            
                            # 根据位置分类
                            spike_info = (cin, i, j, current_input_group[cin, i, j], kernel_add, x, m, y, n)
                            if j < out_w and i < out_h:
                                normal_list.append(spike_info)
                            elif j >= out_w:
                                sub_list.append(spike_info)
                            elif j < out_w and i >= out_h:
                                right_list.append(spike_info)
            
            # 处理pack，每次从三个列表各取一个
            while normal_list or sub_list or right_list:
                # 从三个列表各弹出一个数据
                pack = []
                if normal_list:
                    pack.append(normal_list.pop(0))
                if sub_list:
                    pack.append(sub_list.pop(0))
                if right_list:
                    pack.append(right_list.pop(0))
                
                # 处理pack中的每个spike
                for spike_info in pack:
                    cin, i, j, value, kernel_add, x, m, y, n = spike_info
                    # 累加操作
                    output_block[0, x:m+1, y:n+1] += value * kernel_add
                    
                    # 统计累加次数
                    self.add_num += kernel_add.numel()
                
                # 假设每个pack仅需一拍即可完成计算
                self.cycle_num += 1

        
        # 计算输入块的稀疏度
        total_elements = in_c * in_h * in_w
        non_zero_elements = torch.count_nonzero(input_block).item()
        sparsity = (total_elements - non_zero_elements) / total_elements
        
        # 更新统计信息
        self.stats.num_ops += self.add_num
        self.stats.compute_cycles += self.cycle_num  # 每个spike仅需一拍即可完成计算
        
        return output_block, self.add_num, self.cycle_num, sparsity

class Core:
    """
    核心计算单元，根据加速器配置实现功能
    每个core对应一个Cout，包含4个PE，PE之间并行工作
    """
    def __init__(self, name, cout_idx, num_pes=4):
        self.name = name
        self.cout_idx = cout_idx  # 对应哪个Cout
        self.num_pes = num_pes  # 固定为4个PE
        self.pes = [PE(f"{name}_pe{i}", i) for i in range(num_pes)]
        self.stats = Stats()    # 这个有啥用，为啥要重复统计
        self.total_add_num = 0
        self.total_tile_num = 0
        self.total_cycle_num = 0
        self.compute_cycles_num = 0  # 存储计算周期数
        self.tile_stats = []  # 存储每个tile的统计信息
        self.print_info = True  # 控制是否打印core信息
        
    def configure_accelerator(self, accelerator_config):
        """
        配置加速器参数
        
        Args:
            accelerator_config: 包含加速器配置参数的字典
        """
        self.accelerator_config = accelerator_config
        
    # padding 在core内完成
    def process(self, input_tensor, kernel_tensor, padding=1, stride=1):
        """
        处理整个输入张量和对应的卷积核张量，输出1xHxW的结果（对应一个Cout）
        
        Args:
            input_tensor: HxWxCin的输入张量
            kernel_tensor: 3x3xCin的卷积核张量（对应一个Cout）
            padding: 填充大小
            stride: 步长
            
        Returns:
            output_tensor: 1xHxW的输出张量
            stats: 统计信息
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
        
        self.total_add_num = 0
        self.total_tile_num = 0
        
        # 重置统计信息
        self.stats = Stats()

        # 也重置所有 PE 的统计信息
        for pe in self.pes:
            pe.stats = Stats()
        
        # 创建所有需要处理的tile
        # 在这里做tile时有个技巧，4个PE完成Cin*8*的OF计算，成为stride操作，
        # 那么对padded_input做tile时，也是每个stride操作的输入数据就是Cin * 10 * 10的数据
        # 相较于下面这种直接先行后列的tile方式，RAM的空间会小一点
        tiles = []
        for a in range(0, out_h, 4):
            for b in range(0, out_w, 4):
                # 选取对应的输入块 (Cinx6x6)
                input_tile = padded_input[:, a:a+6, b:b+6]
                tiles.append((input_tile, a, b))
                self.total_tile_num += 1
        
        # stride操作:当tile数目大于PE数目时，4个pe计算一次
        for stride_start in range(0, len(tiles), self.num_pes):
            # 处理当前批次的tile
            stride_end = min(stride_start + self.num_pes, len(tiles))
            stride_tiles = tiles[stride_start:stride_end]   # 4个tile
            
            # 每组4个tile计算前，清空时钟统计信息
            pe_cycles= []
            for i, (input_tile, a, b) in enumerate(stride_tiles):
                pe_idx = i % self.num_pes
                pe = self.pes[pe_idx]     #划分给每个pe一个tile
                
                # PE执行卷积计算
                output_tile, add_num, compute_cycles, sparsity = pe.process(input_tile, kernel_tensor)
                
                # 将结果写入输出矩阵，a和b列，一定要确保写对位置
                output_tensor[0, a:a+4, b:b+4] = output_tile[0, :, :]
                
                # 累加统计信息
                self.total_add_num += add_num
                pe_cycles.append(compute_cycles)
                
                # 记录当前tile的统计信息
                self.tile_stats.append({
                    'tile_position': (a, b),  # tile在输出特征图中的位置
                    'pe_id': pe.pe_id,  # 处理该tile的PE ID
                    'add_num': add_num,  # 累加操作次数
                    'compute_cycles': compute_cycles,  # 计算周期数
                    'sparsity': sparsity,  # 输入块的稀疏度
                    'processing_order': stride_start # 处理顺序,在哪个stride被处理的
                })
        
            # pe的计算周期，由于四个pe是并行的，所以取存储了4个pe的计算周期的pe_cycles数组中的最大周期数
            # print(f"stride:{stride_start}时,PE{pe.pe_id}的计算周期数：", [pe_cycles[i] for i in range(len(pe_cycles))])
            self.total_cycle_num += max(pe_cycles)
            self.compute_cycles_num += max(pe_cycles)

        # 完成了一个Cout的计算，更新self.stats,收集所有PE的统计信息,
        pe_compute_cycles = []
        for pe in self.pes:
            self.stats.num_ops += pe.stats.num_ops   # pe.stats中的信息已经是一直累加的了
            pe_compute_cycles.append(pe.stats.compute_cycles)  
        # 计算总周期数（PE并行工作，所以总周期数等于单个PE的最大周期数）
        self.stats.compute_cycles = max(pe_compute_cycles)
        # todo,等RAM建模后，还需要加上RAM的周期数
        self.stats.total_cycles = max(pe_compute_cycles)
        
        # 打印当前core的统计信息（仅在计算一个Cout时打印）
        if self.print_info:
            print(f"\n=== Core {self.name} (Cout {self.cout_idx}) 统计信息 ===")
            print(f"Total Cycles: {self.stats.total_cycles}")
            print(f"Compute Cycles: {self.stats.compute_cycles}")
            print(f"Total Operations: {self.stats.num_ops}")
            print(f"Total Tiles Processed: {self.total_tile_num}")
            
            # 打印tile统计信息的摘要
            if self.tile_stats:
                print("\nTile Statistics Summary:")
                print(f"Total Tiles Processed: {len(self.tile_stats)}")
                
                # 计算平均和最大/最小计算周期
                compute_cycles_list = [tile['compute_cycles'] for tile in self.tile_stats]
                sparsity_list = [tile['sparsity'] for tile in self.tile_stats]
                
                avg_cycles = sum(compute_cycles_list) / len(compute_cycles_list)
                max_cycles = max(compute_cycles_list)
                min_cycles = min(compute_cycles_list)
                
                avg_sparsity = sum(sparsity_list) / len(sparsity_list)
                max_sparsity = max(sparsity_list)
                min_sparsity = min(sparsity_list)
                
                print(f"Average Compute Cycles per Tile: {avg_cycles:.2f}")
                print(f"Max Compute Cycles: {max_cycles}, Min Compute Cycles: {min_cycles}")
                print(f"Average Sparsity: {avg_sparsity:.2%}, Max Sparsity: {max_sparsity:.2%}, Min Sparsity: {min_sparsity:.2%}")
                
                # 打印每个tile的详细统计信息
                print("\nDetailed Tile Statistics:")
                print("Tile Position | PE ID | Compute Cycles | Sparsity")
                print("-" * 50)
                for tile in self.tile_stats:
                    print(f"({tile['tile_position'][0]}, {tile['tile_position'][1]}) | {tile['pe_id']} | {tile['compute_cycles']} | {tile['sparsity']:.2%}")
        
        return output_tensor, self.stats

class OutProductSimulator:
    """
    外积卷积模拟器，根据加速器配置实现功能
    包含8个core，每个core对应一个Cout，所有core共享相同的input_tensor
    """
    def __init__(self):
        self.num_cores = 8  # 固定为8个core
        self.cores = [Core(f"core{i}", i, num_pes=4) for i in range(self.num_cores)]
        self.global_stats = Stats()
        self.print_core_info = True  # 控制是否打印core信息，只打印一次
        
    def configure_accelerator(self, accelerator_config):
        """
        配置所有核心的加速器参数
        
        Args:
            accelerator_config: 包含加速器配置参数的字典
        """
        for core in self.cores:
            core.configure_accelerator(accelerator_config)
    
    def run_convolution(self, input_tensor, kernel_tensor, padding=1, stride=1):
        """
        执行外积卷积模拟
        
        Args:
            input_tensor: 5D输入张量 [T,B,Cin,H,W]
            kernel_tensor: 4D权重张量 [Cout,Cin,k_h,k_w]
            padding: 填充大小
            stride: 步长
            
        Returns:
            output_tensor: 5D输出张量 [T,B,Cout,H_out,W_out]
            stats: 全局统计信息
        """
        # 确保输入张量是5D，权重张量是4D
        assert len(input_tensor.shape) == 5, f"input_tensor must be 5D, but got {len(input_tensor.shape)}D"
        assert len(kernel_tensor.shape) == 4, f"kernel_tensor must be 4D, but got {len(kernel_tensor.shape)}D"
        
        # 重置全局统计信息
        self.global_stats = Stats()
        
        # 检查输入尺寸
        T, B, in_c, in_h, in_w = input_tensor.shape
        Cout, in_c_kernel, k_h, k_w = kernel_tensor.shape
        
        # 确保Cin一致
        assert in_c == in_c_kernel, f"Cin mismatch: input={in_c}, kernel={in_c_kernel}"
        
        # 计算输出尺寸
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1
        
        # 初始化输出张量 (T,B,Cout,H_out,W_out)
        output_tensor = torch.zeros((T, B, Cout, out_h, out_w), dtype=kernel_tensor.dtype, device=input_tensor.device)
        
        # 对Cout进行拆分，8个Cout为一个group，每个group一起计算，group内每个Cout对应一个core
        # 计算需要多少个group来处理所有Cout
        # 每个group包含8个Cout，对应8个core
        num_groups = Cout // self.num_cores
        # print(f"{Cout}个输出通道,需要分成{num_groups}个group")
        
       
        #tag,每个核处理时，先把所有时间步算完，膜电位就无需再切换，因为batch之间膜电位独立，time step之间膜电位有顺序
        for b in range(B):
            for t in range(T):
                # 按group处理Cout,直到得到一个时间步的完整输出特征图
                print("-batch:", b, "time step:", t,"-")
                group_idx = 0
                    # 获取当前T和B的输入张量 [Cin,H,W]
                # todo,由于没有片上RAM限制，这里假设整个输入张量都可以 fit in memory
                # print(f"------group_idx:{group_idx}------")
                current_input = input_tensor[t, b, :, :]
                # 计算当前group的Cout起始和结束索引
                cout_start = group_idx * self.num_cores
                cout_end = min(cout_start + self.num_cores, Cout)
                
                # 计算当前group实际处理的Cout数量
                current_group_size = cout_end - cout_start
                
                # 每个core对应一个Cout，并行处理当前group内的Cout
                for i in range(current_group_size):
                    cout_idx = cout_start + i
                    # 获取当前Cout对应的卷积核 [Cin,k_h,k_w]
                    current_kernel = kernel_tensor[cout_idx, :, :, :]
                    
                    # 分配到对应的core处理
                    core = self.cores[i]  # 每个group内的Cout按顺序分配给core
                    
                    # 执行卷积计算（所有core共享相同的input_tensor），这个时候的current_input还没有padding
                    # 控制是否打印core信息，只打印一次
                    if self.print_core_info:
                        output_tile, core_stats = core.process(current_input, current_kernel, padding, stride)
                        self.print_core_info = False  # 打印一次后关闭
                    else:
                        # 不打印信息时，暂时修改core的print_info属性
                        # original_print_info = core.print_info
                        core.print_info = False
                        output_tile, core_stats = core.process(current_input, current_kernel, padding, stride)
                        # core.print_info = original_print_info  # 恢复原始值
                    
                    # 将结果写入输出张量
                    output_tensor[t, b, cout_idx, :, :] = output_tile[0, :, :]
                    
                    # 汇总统计信息（core并行工作，周期数都一样）
                    self.global_stats.num_ops += core_stats.num_ops   
                    self.global_stats.num_tiles += core.total_tile_num  # 更新整体tile数量统计，这个信息不重要了
                    group_compute_cycles = core_stats.compute_cycles   
                    group_total_cycles = core_stats.total_cycles

                    #在这里打印core信息也行吧
                # 所有core的计算周期数是一样的,随便取一个就好
                # print(f"group_compute_cycles:{group_compute_cycles},group_total_cycles:{group_total_cycles}")
                #由于每个group的计算过程相同，周期数只需要计算一次然后乘以循环次数
                self.global_stats.total_cycles += group_total_cycles * num_groups
                self.global_stats.compute_cycles += group_compute_cycles * num_groups
                # for group_idx in range(num_groups):
                #      # 获取当前T和B的输入张量 [Cin,H,W]
                #     # todo,由于没有片上RAM限制，这里假设整个输入张量都可以 fit in memory
                #     print(f"--------group_idx:{group_idx}------------")
                #     current_input = input_tensor[t, b, :, :]
                #     # 计算当前group的Cout起始和结束索引
                #     cout_start = group_idx * self.num_cores
                #     cout_end = min(cout_start + self.num_cores, Cout)
                    
                #     # 计算当前group实际处理的Cout数量
                #     current_group_size = cout_end - cout_start
                    
                #     # 每个core对应一个Cout，并行处理当前group内的Cout
                #     for i in range(current_group_size):
                #         cout_idx = cout_start + i
                #         # 获取当前Cout对应的卷积核 [Cin,k_h,k_w]
                #         current_kernel = kernel_tensor[cout_idx, :, :, :]
                        
                #         # 分配到对应的core处理
                #         core = self.cores[i]  # 每个group内的Cout按顺序分配给core
                        
                #         # 执行卷积计算（所有core共享相同的input_tensor），这个时候的current_input还没有padding
                #         # 控制是否打印core信息，只打印一次
                #         if self.print_core_info:
                #             output_tile, core_stats = core.process(current_input, current_kernel, padding, stride)
                #             self.print_core_info = False  # 打印一次后关闭
                #         else:
                #             # 不打印信息时，暂时修改core的print_info属性
                #             # original_print_info = core.print_info
                #             core.print_info = False
                #             output_tile, core_stats = core.process(current_input, current_kernel, padding, stride)
                #             # core.print_info = original_print_info  # 恢复原始值
                        
                #         # 将结果写入输出张量
                #         output_tensor[t, b, cout_idx, :, :] = output_tile[0, :, :]
                        
                #         # 汇总统计信息（core并行工作，周期数都一样）
                #         self.global_stats.num_ops += core_stats.num_ops   
                #         self.global_stats.num_tiles += core.total_tile_num  # 更新整体tile数量统计，这个信息不重要了
                #         group_compute_cycles = core_stats.compute_cycles   
                #         group_total_cycles = core_stats.total_cycles

                #         #在这里打印core信息也行吧
                #     # 所有core的计算周期数是一样的,随便取一个就好
                #     self.global_stats.total_cycles += group_total_cycles
                #     self.global_stats.compute_cycles += group_compute_cycles
        
        # 打印整体统计信息
        # print("\n=== 整体统计信息 ===")
        # print(f"Total Cycles: {self.global_stats.total_cycles}")
        # print(f"Compute Cycles: {self.global_stats.compute_cycles}")
        # print(f"Total Operations: {self.global_stats.num_ops}")
        # print(f"Total Tiles: {self.global_stats.num_tiles}")
        
        return output_tensor, self.global_stats

# 示例用法
if __name__ == "__main__":    

# 测试5D输入和4D权重
    print("\n=== 5D输入和4D权重测试 ===")
    T, B, Cin, H, W = 2, 2, 4, 16, 16  #会有4个stride操作，每个pe复用4个tile，所以总共有16个tile被处理
    Cout = 8
    
    # 创建5D输入张量 [T,B,Cin,H,W]
    input_tensor = torch.zeros(T, B, Cin, H, W)
    torch.manual_seed(42)
    num_ones = int(T * B * Cin * H * W * 0.1)  # 10%的稀疏度
    indices = torch.randperm(T*B*Cin*H*W)[:num_ones]
    input_tensor.view(-1)[indices] = 1
    
    # 创建4D权重张量 [Cout,Cin,k_h,k_w]
    kernel_values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    kernel_tensor = kernel_values.repeat(Cout, Cin, 1, 1)
    
    simulator = OutProductSimulator()
    output_tensor, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)
    
    print(f"模拟器输入形状: {input_tensor.shape}")
    print(f"模拟器输出形状: {output_tensor.shape}")
    print(f"总周期: {stats.total_cycles}")
    print(f"计算周期: {stats.compute_cycles}")
    print(f"总操作数: {stats.num_ops}")
