import torch
import numpy as np
import sys
import os

# 假设utils中有Stats和ceil_a_by_b
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Stats, ceil_a_by_b


class Core:
    """
    每个Core对应一个Cout，直接管理权重和加法树资源。
    """
    def __init__(self, name, cout_idx, num_pes=64, adder_tree_config=None):
        self.name = name
        self.cout_idx = cout_idx
        self.num_pes = num_pes
        self.weights = None 
        if adder_tree_config is None:
            # 默认配置：16：8：8
            self.adder_tree_config = {
                'ranges': [(1,3), (4,6), (7,9)],
                'counts': [16, 8, 8]
            }
        else:
            self.adder_tree_config = adder_tree_config
            
        self.stats = Stats()
        self.compute_cycles_num = 0
        self.adder_insufficient_positions = []

    def configure_core(self, kernel=None):
        """将当前Cout对应的所有通道(Cin)的权重加载到Core的寄存器中"""
        if kernel is not None:
            num_valid = kernel.shape[0]
            # 初始化权重寄存器 [num_pes, 9]
            self.weights = torch.zeros((self.num_pes, 9), dtype=torch.float32, device=kernel.device)
            if num_valid > 0:
                self.weights[:num_valid, :] = kernel[:num_valid, :, :].reshape(num_valid, 9).float()

    # 带有优先级的调度器
    def _hardware_scheduler(self, window_ones_counts):
        """
        核心重构：优先调度T3,再是T2,最后T1，且高级加法树空缺时可处理低级窗口
        因仅统计cycle，采用数学计数方式模拟硬件贪心调度行为。
        window_ones_counts: 长度为 num_pes 的 list，包含每个窗口中1的个数
        """
        cfg_ranges = self.adder_tree_config['ranges']
        LIMIT_T1, LIMIT_T2, LIMIT_T3 = self.adder_tree_config['counts']
        
        # Step 1: 硬件分类统计 (并行比较器)
        mask_t1_sources = []
        mask_t2_sources = []
        mask_t3_sources = []
        
        for i, c in enumerate(window_ones_counts):
            if cfg_ranges[2][0] <= c <= cfg_ranges[2][1]:   # T3优先
                mask_t3_sources.append(i)
            elif cfg_ranges[1][0] <= c <= cfg_ranges[1][1]: # T2其次
                mask_t2_sources.append(i)
            elif cfg_ranges[0][0] <= c <= cfg_ranges[0][1]: # T1最后
                mask_t1_sources.append(i)
        
        # 将其转为队列，用于模拟每拍数据的消耗
        queue_t3 = list(mask_t3_sources)
        queue_t2 = list(mask_t2_sources)
        queue_t1 = list(mask_t1_sources)
        
        current_window_cycle = 0
        
        # 记录每种资源是否发生了 Stalling (不够用)
        #Stall定义：如果某一拍某种加法树由于待处理窗口太多而必须满载运行，且仍有同类或更低类窗口剩下
        stall_t3 = False
        stall_t2 = False
        stall_t1 = False

        # 如果没有窗口需要计算，直接返回0拍
        if not (queue_t3 or queue_t2 or queue_t1):
            return 0, (False, False, False)

        # =====================================================================
        # Step 2 & 3: 并行优先级调度 & 填充 (模拟硬件拍数)
        # =====================================================================
        while queue_t3 or queue_t2 or queue_t1:
            current_window_cycle += 1
            
            # --- 1. T3 加法树调度 (优先级最高，容量 LIMIT_T3) ---
            ports_rem_t3 = LIMIT_T3
            # 优先消耗 T3 窗口
            num_t3_this_cycle = min(len(queue_t3), ports_rem_t3)
            if len(queue_t3) > ports_rem_t3: stall_t3 = True # T3满载且仍有剩余，发生Stall
            queue_t3 = queue_t3[num_t3_this_cycle:]
            ports_rem_t3 -= num_t3_this_cycle
            
            # T3 空缺时填入 T2 窗口
            if ports_rem_t3 > 0 and queue_t2:
                num_t2_by_t3 = min(len(queue_t2), ports_rem_t3)
                queue_t2 = queue_t2[num_t2_by_t3:]
                ports_rem_t3 -= num_t2_by_t3
                
            # T3 扔有空缺时填入 T1 窗口
            if ports_rem_t3 > 0 and queue_t1:
                num_t1_by_t3 = min(len(queue_t1), ports_rem_t3)
                queue_t1 = queue_t1[num_t1_by_t3:]
                ports_rem_t3 -= num_t1_by_t3
                
            # --- 2. T2 加法树调度 (优先级中，容量 LIMIT_T2) ---
            ports_rem_t2 = LIMIT_T2
            # 消耗剩下的 T2 窗口
            num_t2_this_cycle = min(len(queue_t2), ports_rem_t2)
            if len(queue_t2) > ports_rem_t2: stall_t2 = True # T2满载且剩余，Stall
            queue_t2 = queue_t2[num_t2_this_cycle:]
            ports_rem_t2 -= num_t2_this_cycle
            
            # T2 空缺时填入剩下的 T1 窗口
            if ports_rem_t2 > 0 and queue_t1:
                num_t1_by_t2 = min(len(queue_t1), ports_rem_t2)
                queue_t1 = queue_t1[num_t1_by_t2:]
                ports_rem_t2 -= num_t1_by_t2
                
            # --- 3. T1 加法树调度 (优先级最低，容量 LIMIT_T1) ---
            ports_rem_t1 = LIMIT_T1
            # 消耗剩下的 T1 窗口
            num_t1_this_cycle = min(len(queue_t1), ports_rem_t1)
            if len(queue_t1) > ports_rem_t1: stall_t1 = True # T1满载且剩余，Stall
            queue_t1 = queue_t1[num_t1_this_cycle:]
            ports_rem_t1 -= num_t1_this_cycle

        # 对于优先级调度，只要周期 > 1 且发生了 Stall，就记录受限位置
        # (因为 T1 stall 可能导致整体延时)
        has_stall = stall_t3 or stall_t2 or stall_t1
        overflows = (stall_t3, stall_t2, stall_t1) if current_window_cycle > 1 and has_stall else (False, False, False)

        return current_window_cycle, overflows

    # 直接调度
    # def _hardware_scheduler(self, window_ones_counts):
    #         """
    #         模拟硬件调度器：基于并行前缀和，单拍生成所有窗口到物理加法树的路由映射
    #         window_ones_counts: 长度为 num_pes 的 list，包含每个窗口中1的个数
    #         """
    #         num_inputs = len(window_ones_counts)
    #         cfg_ranges = self.adder_tree_config['ranges']
    #         LIMIT_T1, LIMIT_T2, LIMIT_T3 = self.adder_tree_config['counts']
            
    #         # Step 1: 硬件掩码生成 (并行比较器)
    #         mask_t1 = [1 if cfg_ranges[0][0] <= c <= cfg_ranges[0][1] else 0 for c in window_ones_counts]
    #         mask_t2 = [1 if cfg_ranges[1][0] <= c <= cfg_ranges[1][1] else 0 for c in window_ones_counts]
    #         mask_t3 = [1 if cfg_ranges[2][0] <= c <= cfg_ranges[2][1] else 0 for c in window_ones_counts]
    #         # print(f"window num: T1={sum(mask_t1)}, T2={sum(mask_t2)}, T3={sum(mask_t3)}")
            
    #         #todo，待优化，感觉硬件开销大
    #         # Step 2: 硬件并行前缀和计算 (Kogge-Stone Tree),也是LoAS中的prefix-sum电路,计算出每个'1'位之前有多少个'1'
    #         idx_t1 = [sum(mask_t1[:i]) for i in range(num_inputs)]
    #         idx_t2 = [sum(mask_t2[:i]) for i in range(num_inputs)]
    #         idx_t3 = [sum(mask_t3[:i]) for i in range(num_inputs)]
    #         # print(f"idx_t1[-1]={idx_t1[-1]}, idx_t2[-1]={idx_t2[-1]}, idx_t3[-1]={idx_t3[-1]}")
            
    #         # 计算所需的总周期数
    #         sum_t1 = sum(mask_t1)
    #         sum_t2 = sum(mask_t2)
    #         sum_t3 = sum(mask_t3)
    #         max_cycle_t1 = ceil_a_by_b(sum_t1, LIMIT_T1) if sum_t1 > 0 else -1
    #         max_cycle_t2 = ceil_a_by_b(sum_t2, LIMIT_T2) if sum_t2 > 0 else -1
    #         max_cycle_t3 = ceil_a_by_b(sum_t3, LIMIT_T3) if sum_t3 > 0 else -1
    #         # print(f"Max cycle: T1={max_cycle_t1}, T2={max_cycle_t2}, T3={max_cycle_t3}")
            
    #         total_cycles = max(max_cycle_t1, max_cycle_t2, max_cycle_t3) 
    #         # print(f"Total cycles: {total_cycles}")
            
    #         # 记录哪些资源发生了阻塞 (Stall)
    #         overflows = (max_cycle_t1 > 0, max_cycle_t2 > 0, max_cycle_t3 > 0)
            
    #         if total_cycles == 0:
    #             return 0, [], overflows # 全0窗口，0拍跳过
                
    #         schedules = [{'T1': [None]*LIMIT_T1, 'T2': [None]*LIMIT_T2, 'T3': [None]*LIMIT_T3} for _ in range(total_cycles)]
            
    #         # Step 3: MUX目标靶向匹配 (填充路由表)
    #         for i in range(num_inputs):
    #             if mask_t1[i]:
    #                 cycle, port = divmod(idx_t1[i], LIMIT_T1)
    #                 schedules[cycle]['T1'][port] = i  # 记录：通道i接入T1的port端口
    #             elif mask_t2[i]:
    #                 cycle, port = divmod(idx_t2[i], LIMIT_T2)
    #                 schedules[cycle]['T2'][port] = i
    #             elif mask_t3[i]:
    #                 cycle, port = divmod(idx_t3[i], LIMIT_T3)
    #                 schedules[cycle]['T3'][port] = i

    #         return total_cycles, schedules, overflows
    def process(self, input_tensor, kernel_tensor, padding=1, stride=1):
        in_c, in_h, in_w = input_tensor.shape
        k_h, k_w = 3, 3
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1

        padded_input = torch.zeros((in_c, in_h + 2 * padding, in_w + 2 * padding),
                                   dtype=input_tensor.dtype, device=input_tensor.device)
        padded_input[:, padding:padding+in_h, padding:padding+in_w] = input_tensor

        output_tensor = torch.zeros((1, out_h, out_w), dtype=torch.float32, device=input_tensor.device)

        self.compute_cycles_num = 0
        self.stats = Stats()
        self.adder_insufficient_positions = []

        # 遍历每个输出特征图位置 (空间迭代)
        for h in range(out_h):
            for w in range(out_w):
                h_start, w_start = h * stride, w * stride
                h_end, w_end = h_start + k_h, w_start + k_w
                input_tile = padded_input[:, h_start:h_end, w_start:w_end] # [Cin, 3, 3]

                # =====================================================================
                # Pipeline Stage 1: Hardware Scheduler (单拍生成路由表)
                # =====================================================================
                # 利用PyTorch加速计算64个窗口中非零元素的个数
                one_nums = torch.sum(input_tile, dim=(1, 2)).int().tolist()
                
                # # 调用优化后的优先级调度算法
                # total_cycles, overflows = self._hardware_scheduler(one_nums)
                total_cycles, schedules, overflows = self._hardware_scheduler(one_nums)

                if total_cycles > 0:
                    self.compute_cycles_num += total_cycles

                # 记录Stall溢出情况
                if total_cycles > 1:
                    # 如果任何一级加法树发生了满载导致 Stalling，我们记录类型
                    # (由于 T3 可处理 T1/T2，T1 发生 stall 的概率会显著降低)
                    if overflows[0]: self.adder_insufficient_positions.append((h, w, 0)) # 受限于 T3
                    if overflows[1]: self.adder_insufficient_positions.append((h, w, 1)) # 受限于 T2
                    if overflows[2]: self.adder_insufficient_positions.append((h, w, 2)) # 受限于 T1

                # =====================================================================
                # Pipeline Stage 2 & 3: Local Wallace Tree & Global Reduction
                # 仅统计cycle，所以直接注释
                # =====================================================================
                '''
                spatial_psum = 0.0 
                output_tensor[0, h, w] = spatial_psum
                '''

        # 增加固定流水线延迟 (1拍取数/调度 + 1拍全局累加写回)
        total_pipeline_cycles = self.compute_cycles_num + 2   
        
        ideal_cycles = out_h * out_w  
        print(f'理想计算{ideal_cycles}拍，实际{total_pipeline_cycles}拍 (加速/压缩率: {total_pipeline_cycles/ideal_cycles:.2f})')
        
        self.stats.compute_cycles += self.compute_cycles_num
        return output_tensor, total_pipeline_cycles


class InnerProductSimulator:
    """
    外积卷积模拟器，包含多个Core，每个Core对应一个Cout
    """
    def __init__(self, num_cores=16, num_pes=64):
        # 配置加法树的数目为 16：8：8
        self.adder_tree_config = {
            'ranges': [(1,3), (4,6), (7,9)],
            'counts': [16, 8, 8]
        }
        
        self.num_cores = num_cores
        self.num_pes = num_pes
        self.RAM_size = 32*32*64
        self.bandwidth = 32
        self.weight_size = 16
        self.mp_size = 32
        self.lif_num = 1
        self.processed_cin = self.num_pes

        self.cores = [Core(f"core{i}", i, self.num_pes, self.adder_tree_config) for i in range(self.num_cores)]
        self.global_stats = Stats()

        # 数据搬运统计
        self.load_kernel_cycle = 0
        self.load_activation_cycle = 0
        self.read_mp_cycle = 0
        self.write_activation_cycle = 0
        self.all_adder_insufficient_positions = []

    def configure_accelerator(self, kernel=None):
        """配置所有核心的权重"""
        for i, core in enumerate(self.cores):
            core_kernel = kernel[i, :, :, :] if kernel is not None else None
            core.configure_core(core_kernel)

    def run_convolution(self, input_tensor, kernel_tensor, padding=1, stride=1):
        """
        input_tensor: [T, B, Cin, H, W] 或 [T, Cin, H, W] (B=1)
        kernel_tensor: [Cout, Cin, 3, 3]
        """
        self.global_stats = Stats()
        self.load_kernel_cycle = 0
        self.load_activation_cycle = 0
        self.compute_cycles = 0
        self.total_cycles = 0

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
        output_tensor = torch.zeros((T, B, Cout, out_h, out_w), dtype=kernel_tensor.dtype, device=input_tensor.device)

        Cout_group = ceil_a_by_b(Cout, self.num_cores)

        for b in range(B):
            print(f"Processing batch {b}")
            core = self.cores[0]
            channel_operation = ceil_a_by_b(in_c, self.processed_cin)
            
            for cin_group_idx in range(channel_operation):
                cin_start = cin_group_idx * self.processed_cin
                cin_end = min(cin_start + self.processed_cin, in_c)

                kernel_data = 1 * (cin_end - cin_start) * k_h * k_w * self.weight_size
                self.load_kernel_cycle += kernel_data / self.bandwidth
                self.global_stats.reads['dram'] += kernel_data
                self.global_stats.data_moved['kernel'] += kernel_data

                core_kernel = kernel_tensor[0, cin_start:cin_end, :, :]
                core.configure_core(core_kernel)

                for t in range(T):
                    act_data = (cin_end - cin_start) * in_h * in_w
                    self.load_activation_cycle = act_data / self.bandwidth
                    self.global_stats.reads['dram'] += act_data
                    self.global_stats.data_moved['act'] += act_data

                    current_input = input_tensor[t, b, cin_start:cin_end, :, :]

                    out_tile, cycle = core.process(current_input, kernel_tensor[0, cin_start:cin_end, :, :],
                                                    padding, stride)
                    # output_tensor[t, b, 0, :, :] += out_tile[0, :, :]

                    self.compute_cycles += cycle
                    self.total_cycles += max(cycle, self.load_activation_cycle)

            self.all_adder_insufficient_positions.extend(core.adder_insufficient_positions)
            lif_cycle = out_h * out_w * T / self.lif_num
            self.total_cycles = max(self.total_cycles, lif_cycle)

        activation_write_data = Cout * out_h * out_w
        self.write_activation_cycle = activation_write_data / self.bandwidth
        self.global_stats.writes['dram'] += activation_write_data
        self.global_stats.data_moved['act'] += activation_write_data

        self.global_stats.total_cycles = self.total_cycles * Cout_group
        self.global_stats.compute_cycles = self.compute_cycles * Cout_group

        return output_tensor, self.global_stats


if __name__ == "__main__":
    # 测试
    T, B, Cin, H, W = 1, 1, 64, 8, 8
    Cout = 16
    input_tensor = torch.randint(0, 2, (T, B, Cin, H, W), dtype=torch.float32)
    kernel_tensor = torch.randn(Cout, Cin, 3, 3)

    simulator = InnerProductSimulator(num_cores=16, num_pes=64)
    output, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)
    print(f"Output shape: {output.shape}")
    print(f"Compute cycles: {stats.compute_cycles}")
    print(f"Total cycles: {stats.total_cycles}")
    print(f"Adder tree config: {simulator.adder_tree_config}")
