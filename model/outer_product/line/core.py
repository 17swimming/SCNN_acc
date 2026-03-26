"""
先按照一个core处理一个cin来设计

输入：if（[H,W]的0/1输入特征图），cin（指明是哪个输入通道）

SplitUnit从if中取一行（行号为r），输出bitstream（此时索引为0到len(bitstream)），c，r。


权重配置：根据cin，取卷积核；3*3卷积核中9个权重编号分别为0-8，012放入PE并命名为W0，345放入PE并命名为W1，678放入PE并命名为W2，这三个PE合称PU。一个核内放16个PU，分别对应不同的Cout。

移位寄存器：bitstream（此时最右侧为索引0）每个cycle逐bit 右移，c队列（此时最右侧为索引0）也跟着右移，依次经过W2,W1,W0。截取3bit的bitstream以及c输入PE。待整个bitstream从PU中滑出时，完成卷积计算。

门控单元规则：
- 如果r=0，关断w2，w1
- 如果r=1，关断w2
- 如果r=h-2，关断w0
- 如果r=h-1，关断w0和w1

"""

import torch
# from model.outer_product.line.shift import SplitUnit
# from model.outer_product.line.PE import PE
from shift import SplitUnit
from PE import PE

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
            pe_w0 = PE()
            pe_w1 = PE()
            pe_W2 = PE()
            self.pus.append({'W0': pe_w0, 'W1': pe_w1, 'W2': pe_W2})
        
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
        :return: 输出特征图
        """
        
        self.cycles = 0  # 归零
        H,W = if_map.shape

        # 初始化输出tensor，应该是[cout, oh, W]
        output_tensors = torch.zeros((cout, oh, ow), dtype=torch.float32)
        
        # 初始化移位寄存器
        shift_reg = torch.zeros(9, dtype=torch.int32)
        c_reg = torch.full((9,), -1, dtype=torch.int32)
        
        # 逐行处理
        for r in range(H):
            # 重置PE的Psum
            for pu in self.pus:
                pu['W0'].reset()
                pu['W1'].reset()
                pu['W2'].reset()
            
            # 使用SplitUnit处理当前行
            bitstream, (_, c_array) = self.split_unit.process(if_map[r], r)
            
            # 计算门控信号
            gate_w0 = 1
            gate_w1 = 1
            gate_W2 = 1
            if r == 0:
                gate_w1 = 0
                gate_W2 = 0
            elif r == 1:
                gate_W2 = 0
            elif r == H-2:
                gate_w0 = 0
            elif r == H-1:
                gate_w0 = 0
                gate_w1 = 0
            
            # 处理当前行的bitstream
            for i in range(len(c_array) + 8):  # i=8时才开始计算第一个值,
                self.cycles += 1
                # 移位寄存器右移（向索引增大的方向）
                shift_reg = torch.roll(shift_reg, shifts=1, dims=0)
                c_reg = torch.roll(c_reg, shifts=1, dims=0)
                
                # 从右侧（索引0）填入新数据
                # 这样经过8拍移位后，bitstream[0]会到达shift_reg[8]
                if i < len(bitstream):
                    shift_reg[0] = bitstream[i]
                else:
                    shift_reg[0] = 0
                
                # 最重要的就是c队列，一定不能错
                if i < len(c_array):
                    c_reg[0] = c_array[i]
                else:
                    c_reg[0] = -1
                
                # 当寄存器中有有效数据时，进行计算                    
                # 遍历被激活的cout个PU进行计算
                for pu_idx in range(cout):
                    pu = self.pus[pu_idx]
                    # 根据门控信号选择要使用的PE
                    if gate_w0:
                        # PE W0对应最右边的3bit (shift_reg[6:9])
                        pu['W0'].process(shift_reg[6:9], c_reg[8])
                    if gate_w1:
                        pu['W1'].process(shift_reg[3:6], c_reg[5])
                    if gate_W2:
                        pu['W2'].process(shift_reg[:3], c_reg[2])
                            
            
            # 收集所有PU的输出，不同的PU对应不同的Cout,
            # 因为只用了cout个PU，所以这里只取cout个PU的输出
            for pu_idx in range(cout):
                if gate_w0:
                    # 每个PE的最大输出宽度为32，根据实际输入特征图的大小w截取
                    output_tensors[pu_idx][r] += self.pus[pu_idx]['W0'].get_psum(ow)
                if gate_w1:
                    # 每个PE的最大输出宽度为32，根据实际输入特征图的大小w截取
                    output_tensors[pu_idx][r-1] += self.pus[pu_idx]['W1'].get_psum(ow)
                if gate_W2:
                    output_tensors[pu_idx][r-2] += self.pus[pu_idx]['W2'].get_psum(ow)  
        
        return output_tensors



