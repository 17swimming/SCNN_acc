import torch
import numpy as np
from collections import OrderedDict
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import Stats, ceil_a_by_b

class PE:
    """
    处理元素(PE)，按照加速器配置实现功能
    每个PE处理一块4x4的输出特征图，每次只能计算一个Cin，因此需要串行处理整个块
    """
    #每次调用 pe.process() 时， add_num 和 cycle_num 只反映当前 tile 的统计信息
    #pe.stats 则反映该 PE 自创建以来处理所有 tile 的累计统计信息
    def __init__(self, name, pe_id, k=2):
        self.name = name
        self.pe_id = pe_id
        self.stats = Stats()   
        self.add_num = 0
        self.cycle_num = 0
        self.k = k   #一次处理k个cin

    def process(self, cube, Psum, kernel_tensor, i, j, a, b, c, d):
        """
        根据cube中1的位置，取kernel_tensor中对应输入通道的权重，与Psum累加
        同时，Psum是裁剪后的，因此需要根据a、b、c、d去取kernel_add，再累加
        kernel_add的大小就是加法次数，每个非零元素需要1个周期

        
        Args:
            cube: Cinx1x1的输入张量块
            Psum: 3x3的部分和张量
            kernel_tensor: CoutxCinx3x3的卷积核张量
            i, j: 当前处理位置的坐标
            a, b, c, d: 卷积核裁剪的边界
            
        Returns:
            Psum: 更新后的部分和张量
            add_num: 累加操作次数
            compute_cycles: 计算周期数
        """
        # 获取cube中1的索引（非零元素的位置）
        nonzero_indices = torch.nonzero(cube).flatten()
        
        # 如果没有非零元素，直接返回
        if len(nonzero_indices) == 0:
            return Psum, 0, 0
        
        # 初始化当前cube的操作数
        self.add_num = 0
        self.cycle_num = 0        
        # 遍历每个非零元素
        for cin_idx in nonzero_indices:
            # 读取kernel_tensor中对应输入通道的权重 [3, 3]
            current_kernel = kernel_tensor[cin_idx, :, :]
            
            # 根据a、b、c、d去取kernel_add
            kernel_add = current_kernel[a:b+1, c:d+1]
            
            # 与Psum累加
            Psum += kernel_add
            
            # 统计累加操作次数
            self.add_num += kernel_add.numel()
        
        # 计算周期数（每个非零元素需要1个周期）
        self.cycle_num = len(nonzero_indices)
        
        # 计算利用率（公式：利用率=add_num/(cycle_num * 9)）
        if self.cycle_num > 0:
            utilization = self.add_num / (self.cycle_num * 9)
            # 确保利用率在0-1范围内
            utilization = min(max(utilization, 0), 1)
        else:
            utilization = 0
        
        # 更新统计信息，持续统计
        self.stats.num_ops += self.add_num
        self.stats.compute_cycles += self.cycle_num  
        self.stats.num_cube += 1
        # 每次处理完一个cube后，看一下这三个参数的情况
        # if(self.pe_id == 0):
        #     print(f"PE {self.pe_id} 在处理i:{i},j:{j}，总计处理了 {self.stats.num_ops } 个操作，{self.stats.compute_cycles} 个周期，{self.stats.num_cube} 个cube")


        return Psum, self.add_num, self.cycle_num, utilization
        

class Core:
    """
    核心计算单元，根据加速器配置实现功能
    每个core对应一个Cout，包含4个PE，PE之间并行工作
    """
    def __init__(self, name, cout_idx, num_pes=4, k=2):
        self.name = name
        self.cout_idx = cout_idx  # 对应哪个Cout
        self.num_pes = num_pes  # 固定为4个PE
        self.pes = [PE(f"{name}_pe{i}", i, k) for i in range(num_pes)]
        self.stats = Stats()    # 这个有啥用，为啥要重复统计
        self.total_add_num = 0
        self.total_tile_num = 0
        self.total_cycle_num = 0
        self.total_cube_num = 0  # 存储需要计算的cube数目
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
            utilization_map: 二维利用率统计图
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
        
        # 初始化利用率统计地图
        utilization_map = torch.zeros((in_h + 2 * padding, in_w + 2 * padding), dtype=torch.float32)
        utilization_count = torch.zeros((in_h + 2 * padding, in_w + 2 * padding), dtype=torch.int32)
        
        self.total_add_num = 0
        self.total_tile_num = 0
        
        # 重置统计信息
        self.stats = Stats()

        # 也重置所有 PE 的统计信息
        for pe in self.pes:
            pe.stats = Stats()
        
        # -----------方法2：  膜电位使用spm，一次处理cin*1*1的cube---------
        # h * w个膜电位存在片上，        
        # 每次取pe_num个cube（in_c*1*1），根据cube中1所在的cin，取对应cin的9个weight
        # 每个pe接收一个cube、kernel、还有9个Psum，即以（i，j）为右下角的3*3的output_tensor
        # 先把整个特征图按照pe数目进行tile，得到h*w/pe_num个tile，
        # 将特征图的每个位置作为一个cube（Cin*1*1），按照PE数量进行分组处理
        # 由于使用了全零跳过机制，所以使用in_h*in_w而不是out_h*out_w
        total_positions = in_h * in_w  # 总位置数
        num_pes = self.num_pes  # PE数量
        
        # 预计算所有非零任务
        tasks = []
        for pos_idx in range(total_positions):
            # 计算当前位置在特征图中的坐标
            i = pos_idx // in_w  # 行索引
            j = pos_idx % in_w   # 列索引
            
            # 获取当前位置的cube（Cin*1*1）
            cube = padded_input[:, i, j]  # [Cin]
            
            # 跳过全0的cube，直接处理下一个
            if torch.all(cube == 0):
                continue
            
            # 获取对应的Psum区域（3x3的输出区域）
            # 确定Psum的有效区域边界
            # 对于输入位置(i,j)，它会影响输出特征图中以(i,j)为右下角的3x3区域
            # 确定卷积核旋转后的有效区域
            a = max(0, 2 - i)
            b = 2 if (i < out_h) else 5 - i
            c = max(0, 2 - j)
            d = 2 if (j < out_w) else 5 - j
            
            # 确定有效区域在输出矩阵中的位置
            m = i if (i < out_h) else out_h - 1
            n = j if (j < out_w) else out_w - 1
            x = m - (b - a)
            y = n - (d - c)

            # 获取Psum区域
            Psum = output_tensor[0, x:m+1, y:n+1]
            
            # 将任务信息加入列表
            tasks.append((cube, Psum, kernel_tensor, i, j, a, b, c, d, x, m, y, n))
        
        # 导入多线程模块
        import threading
        
        # 创建锁，用于保护共享资源
        lock = threading.Lock()
        # 创建打印锁，用于同步打印操作
        print_lock = threading.Lock()
        
        # 计算每个PE应该处理的任务数量
        total_tasks = len(tasks)
        self.total_cube_num = total_tasks
        # with print_lock:
        #     print(f"Core {self.name} (Cout {self.cout_idx}) 需要计算的cube数目: {total_tasks}")
        
        # 为每个PE分配任务（轮询方式，确保任务分布更均匀）
        pe_tasks = [[] for _ in range(num_pes)]
        for task_idx, task in enumerate(tasks):
            # 轮询分配任务给PE
            pe_idx = task_idx % num_pes
            pe_tasks[pe_idx].append(task)
        
        # 存储每个PE的计算周期
        pe_compute_cycles = {pe.pe_id: 0 for pe in self.pes}
        
        # 定义PE工作函数
        def pe_worker(pe, pe_tasks_list):
            total_compute_cycles = 0
            for task in pe_tasks_list:
                try:
                    # 获取任务信息
                    cube, Psum, kernel_tensor, i, j, a, b, c, d, x, m, y, n = task
                    
                    # PE执行计算：根据cube中1的位置，取对应权重，与Psum累加
                    # 传递kernel裁剪信息给PE
                    updated_Psum, add_num, compute_cycles, utilization = pe.process(
                        cube, Psum, kernel_tensor, 
                        i, j, a, b, c, d
                    )
                    
                    # 直接使用从PE.process获取的利用率值
                    pass
                    
                    # 使用锁保护共享资源的访问
                    with lock:
                        # 将更新后的Psum写回输出张量
                        output_tensor[0, x:m+1, y:n+1] = updated_Psum
                        
                        # 累加统计信息
                        self.total_add_num += add_num
                        self.total_tile_num += 1
                        total_compute_cycles += compute_cycles
                        
                        # 更新利用率地图（使用实际的输出位置m,n）
                        if m < in_h + 2 * padding and n < in_w + 2 * padding:
                            utilization_map[m, n] += utilization
                            utilization_count[m, n] += 1
                except Exception as e:
                    print(f"PE {pe.pe_id} 处理任务时出错: {e}")
                    continue
            
            # 任务完成后更新PE的总计算周期
            pe_compute_cycles[pe.pe_id] = total_compute_cycles
        
        # 创建并启动PE线程
        threads = []
        for pe_idx, pe in enumerate(self.pes):
            if pe_idx < len(pe_tasks):
                t = threading.Thread(target=pe_worker, args=(pe, pe_tasks[pe_idx]))
                threads.append(t)
                t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 完成了一个Cout的计算，更新self.stats,收集所有PE的统计信息,
        for pe in self.pes:
            self.stats.num_ops += pe.stats.num_ops   # pe.stats中的信息已经是一直累加的了
        # 从字典中获取每个PE的计算周期
        pe_cycles_list = list(pe_compute_cycles.values())
        # 计算总周期数（PE并行工作，所以总周期数等于单个PE的最大周期数）
        self.stats.compute_cycles = max(pe_cycles_list)
        # todo,等RAM建模后，还需要加上RAM的周期数
        self.stats.total_cycles = max(pe_cycles_list)
        
        # 计算平均利用率
        with lock:
            # 避免除零错误
            mask = utilization_count > 0
            if torch.any(mask):
                utilization_map[mask] /= utilization_count[mask]
        
        # 打印每个PE的统计信息
        if self.print_info:
            with print_lock:
                print(f"\n=== Core {self.name} (Cout {self.cout_idx}) PE统计信息 ===")
                print(f"Total Cycles: {self.stats.total_cycles}")
                print(f"Compute Cycles: {self.stats.compute_cycles}")
                print(f"Total Operations: {self.stats.num_ops}")
                print(f"Total Cubes Processed: {self.total_cube_num}")
                print("PE ID | Cube Count | Total Execution Cycles")
                print("-" * 45)
                for pe in self.pes:
                    print(f"{pe.pe_id:5} | {pe.stats.num_cube:10} | {pe.stats.compute_cycles:20}")
                
        return output_tensor, self.stats, utilization_map.cpu().numpy()

   

class OutProductSimulator:
    """
    外积卷积模拟器，根据加速器配置实现功能
    包含8个core，每个core对应一个Cout，所有core共享相同的input_tensor
    """
    def __init__(self, k=2, num_pes=4):
        self.num_cores = 8  # 固定为8个core
        self.num_pes = num_pes  # 存储每个core的PE数目
        self.k = k  # 存储k值
        self.cores = [Core(f"core{i}", i, num_pes=num_pes, k=k) for i in range(self.num_cores)]
        self._current_num_pes = num_pes  # 存储当前使用的num_pes值
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
        
        # 检查num_pes是否发生变化，如果变化了就重新创建Core实例
        if not hasattr(self, '_current_num_pes') or self._current_num_pes != self.num_pes:
            # 重新创建Core实例，使用新的num_pes值
            self.cores = [Core(f"core{i}", i, num_pes=self.num_pes, k=self.k) for i in range(self.num_cores)]
            self._current_num_pes = self.num_pes
        
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
        
        # 初始化总利用率地图
        total_utilization_map = None
        total_utilization_count = None
        
        # 对Cout进行拆分，8个Cout为一个group，每个group一起计算，group内每个Cout对应一个core
        # 计算需要多少个group来处理所有Cout
        # 每个group包含8个Cout，对应8个core
        num_groups = Cout // self.num_cores
        print(f"{Cout}个输出通道,需要分成{num_groups}个group")
        
       
        #tag,每个核处理时，先把所有时间步算完，膜电位就无需再切换，因为batch之间膜电位独立，time step之间膜电位有顺序
        for b in range(B):
            for t in range(T):
                # tag,没有精确计算，仅计算group0，用于性能评估
                print("-batch:", b, "-time step:", t,"-","only group0")
                group_idx = 0
                # 获取当前T和B的输入张量 [Cin,H,W]
                # todo,由于没有片上RAM限制，这里假设整个输入张量都可以 fit in memory
                # print(f"------group_idx:{group_idx}------")
                current_input = input_tensor[t, b, :, :, :]
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
                        output_tile, core_stats, utilization_map = core.process(current_input, current_kernel, padding, stride)
                        self.print_core_info = False  # 打印一次后关闭
                    else:
                        # 不打印信息时，暂时修改core的print_info属性
                        # original_print_info = core.print_info
                        core.print_info = False
                        output_tile, core_stats, utilization_map = core.process(current_input, current_kernel, padding, stride)
                        # core.print_info = original_print_info  # 恢复原始值
                    
                    # 累加利用率地图（不同t和b的结果累加）
                    if total_utilization_map is None:
                        # 初始化总利用率地图
                        total_utilization_map = np.zeros_like(utilization_map)
                        total_utilization_count = np.zeros_like(utilization_map)
                    total_utilization_map += utilization_map
                    total_utilization_count += np.ones_like(utilization_map)
                    
                    # 将结果写入输出张量
                    output_tensor[t, b, cout_idx, :, :] = output_tile[0, :, :]
                    
                    # 汇总统计信息（core并行工作，周期数都一样）
                    self.global_stats.num_ops += core_stats.num_ops   
                    self.global_stats.num_tiles += core.total_tile_num  # 更新整体tile数量统计，这个信息不重要了
                    # 所有core的cycyle 都一样
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
        
        # 计算平均利用率（考虑不同t和b的累加）
        mask = total_utilization_count > 0
        avg_utilization_map = np.zeros_like(total_utilization_map)
        avg_utilization_map[mask] = total_utilization_map[mask] / total_utilization_count[mask]
        
        # 打印整体统计信息
        # print("\n=== 整体统计信息 ===")
        # print(f"Total Cycles: {self.global_stats.total_cycles}")
        # print(f"Compute Cycles: {self.global_stats.compute_cycles}")
        # print(f"Total Operations: {self.global_stats.num_ops}")
        # print(f"Total Tiles: {self.global_stats.num_tiles}")
        
        return output_tensor, self.global_stats, avg_utilization_map

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
    
    simulator = OutProductSimulator()
    simulator.num_pes = 8
    simulator.num_cores = 16
    simulator.print_core_info = True 
    output_tensor, stats = simulator.run_convolution(input_tensor, kernel_tensor, padding=1, stride=1)

    # 我理论估计我需要的计算周期数
    theorical_cycles = H * W / simulator.num_pes * T * B * math.ceil(Cout/simulator.num_cores) * Cin * dense
    # 稠密计算，
    latency_estimation_nvdla = H * W * 9  * T * B * math.ceil(Cout/16) * math.ceil(Cin/64) 

    
    print(f"模拟器输入形状: {input_tensor.shape}")
    print(f"模拟器输出形状: {output_tensor.shape}")
    print(f"模拟器总周期: {stats.total_cycles}")
    print(f"理论周期: {theorical_cycles}")
    print(f"dla的周期: {latency_estimation_nvdla}")
    print(f"加速比：{latency_estimation_nvdla / stats.total_cycles:.2f}")
