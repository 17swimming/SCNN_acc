"""

"""

import torch
# from shift_copy import SplitUnit
from shift_nop import SplitUnit
from PE_copy import PE

class Core:
    def __init__(self, kernel_size=3, num_pus=16):
        """
        初始化Core
        :param kernel_size: 卷积核大小，默认为3
        :param num_pus: 核内PU数量，默认为16
        """
        self.kernel_size = kernel_size
        self.num_pus = num_pus
        self.cycles = 0  # 处理if_map[H,W]所需cycle数
        self.split_unit = SplitUnit(kernel_size=kernel_size)
        # 初始化16个PU，每个PU包含3个PE（W0, W1, W2）
        self.pus = []
        for _ in range(num_pus):
            # 每个PU包含三个PE，分别对应W0, W1, W2
            pe_w0 = PE(name='W0')
            pe_w1 = PE(name='W1')
            pe_w2 = PE(name='W2')
            self.pus.append({'W0': pe_w0, 'W1': pe_w1, 'W2': pe_w2})
        # PE cycle统计
        self.pe_cycles = torch.zeros(3)  # W0, W1, W2的有效cycle数
        
    def configure_weights(self, cin, kernel):
        """
        配置权重
        :param cin: 输入通道
        :param kernel: 卷积核权重，形状为 [cout, cin, k, k]
        """
        # 遍历每个PU（对应不同的Cout）
        for i in range(min(self.num_pus, kernel.shape[0])):
            # 提取当前PU对应的权重
            pu_kernel = kernel[i, cin, :, :]
            # 按照要求分割权重：012->W0, 345->W1, 678->W2
            # 逆序放入
            weights_w0 = pu_kernel.view(-1)[:3].flip(0)  # 2,1,0
            weights_w1 = pu_kernel.view(-1)[3:6].flip(0)  # 5,4,3
            weights_W2 = pu_kernel.view(-1)[6:9].flip(0)  # 8,7,6
            # 配置PE权重
            self.pus[i]['W0'].set_weights(weights_w0)
            self.pus[i]['W1'].set_weights(weights_w1)
            self.pus[i]['W2'].set_weights(weights_W2)
    
    def process(self, if_map, cout, cin, oh, ow):
        """
        处理输入特征图
        :param if_map: 输入特征图，形状为 [H,W], 已经是padding后的
        :param cin: 指明是哪个输入通道
        :param cout: 指明输出通道数目（不能超过16）
        :return: 处理所需的cycle数和每个PE的有效cycle数
        """
        
        self.cycles = 0  # 归零
        self.pe_cycles = torch.zeros(3)  # 重置PE cycle统计
        H,W = if_map.shape

        # 初始化输出tensor，应该是[cout, oh, W]
        output_tensors = torch.zeros((cout, oh, ow), dtype=torch.float32)
        
        # 初始化PU的输出
        for pu_idx in range(cout):
            pu = self.pus[pu_idx]
            # 确保PE的psum大小正确
            pu['W0'].set_output(output_tensors[pu_idx:pu_idx+1], pu_idx)
            pu['W1'].set_output(output_tensors[pu_idx:pu_idx+1], pu_idx)
            pu['W2'].set_output(output_tensors[pu_idx:pu_idx+1], pu_idx)
        
        # 初始化移位寄存器
        shift_reg = torch.zeros(9, dtype=torch.int32)
        c_reg = torch.full((9,), -1, dtype=torch.int32)
        r_reg = torch.full((9,), -1, dtype=torch.int32)

        # 先经过split_unit，获取bitstream和c_array
        bitstream, r_array, c_array = self.split_unit.process(if_map)
        # print(f"len(c_array) = {len(c_array)}")
        # print(f"bitstream: {bitstream}")
        # print(f"r_array: {r_array}")
        # print(f"c_array: {c_array}")
        # 然后逐cycle 将bitstream流入pe,由于门控机制，整个bitstream流入后，计算就已经结束了。
        # 因为即使最后三行的bitstream只有3b，此时H-1行在w2，H-2行在w1，H-3行在w0，
        # 下一拍w2、w1、w0都输出结果，且后续w1和w都进入门控状态，计算已完成。
        for i in range(len(bitstream)+1):  
            self.cycles += 1
            # 移位寄存器右移（向索引增大的方向）
            shift_reg = torch.roll(shift_reg, shifts=1, dims=0)
            c_reg = torch.roll(c_reg, shifts=1, dims=0)
            r_reg = torch.roll(r_reg, shifts=1, dims=0)
            
            # 从右侧（索引0）填入新数据
            # 这样经过8拍移位后，bitstream[0]会到达shift_reg[8]
            if i < len(bitstream):
                shift_reg[0] = bitstream[i]
            else:
                shift_reg[0] = 0
            
            # 最重要的就是c队列，一定不能错
            if i < len(c_array):
                c_reg[0] = c_array[i]
                r_reg[0] = r_array[i]
            else:
                c_reg[0] = -1
                r_reg[0] = -1

            # 遍历被激活的cout个PU进行计算
            for pu_idx in range(cout):
                pu = self.pus[pu_idx]

                # 调试信息
                # if i >= 8 and i < 15:
                #     print(f"Cycle {i}: shift_reg={shift_reg}, c_reg[8]={c_reg[8]}, r_reg[8]={r_reg[8]}")

                r0, psum0 = pu['W0'].process(shift_reg[6:9], c_reg[8], r_reg[8], H)
                r1, psum1 = pu['W1'].process(shift_reg[3:6], c_reg[5], r_reg[5], H)
                r2, psum2 = pu['W2'].process(shift_reg[:3], c_reg[2], r_reg[2], H)

                # 统计PE的有效cycle数
                # 对于W0：有效的cycle数为r≠-1、H-1，H-2
                if r_reg[8] != -1 and r_reg[8] != H-1 and r_reg[8] != H-2:
                    self.pe_cycles[0] += 1
                # 对于W1：有效的cycle数为r≠-1、0，H-1
                if r_reg[5] != -1 and r_reg[5] != 0 and r_reg[5] != H-1:
                    self.pe_cycles[1] += 1
                # 对于W2：有效的cycle数为r≠-1、0、1
                if r_reg[2] != -1 and r_reg[2] != 0 and r_reg[2] != 1:
                    self.pe_cycles[2] += 1

                # 检查r0 == r1 == r2且对应psum非全零
                if r0 != -1 and r1 != -1 and r2 != -1 and r0 == r1 == r2:
                    psum0_nonzero = not torch.all(psum0 == 0)
                    psum1_nonzero = not torch.all(psum1 == 0)
                    psum2_nonzero = not torch.all(psum2 == 0)
                    if psum0_nonzero and psum1_nonzero and psum2_nonzero:
                        print(f"[WARNING] Cycle {i}: r0 == r1 == r2 == {r0}, psum0={psum0}, psum1={psum1}, psum2={psum2}")

                # 监测r0, r1, r2相同的cycle
                # 检查r0 == r1且对应psum非全零
                elif r0 != -1 and r1 != -1 and r0 == r1:
                    psum0_nonzero = not torch.all(psum0 == 0)
                    psum1_nonzero = not torch.all(psum1 == 0)
                    if psum0_nonzero and psum1_nonzero:
                        print(f"[WARNING] Cycle {i}: r0 == r1 == {r0}, psum0={psum0}, psum1={psum1}")
                
                # 检查r0 == r2且对应psum非全零
                elif r0 != -1 and r2 != -1 and r0 == r2:
                    psum0_nonzero = not torch.all(psum0 == 0)
                    psum2_nonzero = not torch.all(psum2 == 0)
                    if psum0_nonzero and psum2_nonzero:
                        print(f"[WARNING] Cycle {i}: r0 == r2 == {r0}, psum0={psum0}, psum2={psum2}")
                
                # 检查r1 == r2且对应psum非全零
                elif r1 != -1 and r2 != -1 and r1 == r2:
                    psum1_nonzero = not torch.all(psum1 == 0)
                    psum2_nonzero = not torch.all(psum2 == 0)
                    if psum1_nonzero and psum2_nonzero:
                        print(f"[WARNING] Cycle {i}: r1 == r2 == {r1}, psum1={psum1}, psum2={psum2}")
                

                # 根据r0判断是否需要写入
                if(r0 != -1):
                    output_tensors[pu_idx][r0] += psum0
                
                # 根据r1判断是否需要写入
                if(r1 != -1):
                    output_tensors[pu_idx][r1] += psum1
                
                # 根据r2判断是否需要写入
                if(r2 != -1):
                    output_tensors[pu_idx][r2] += psum2

        
        return output_tensors,self.cycles,self.pe_cycles



