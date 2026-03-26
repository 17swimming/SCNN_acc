import torch

class PE:
    def __init__(self, name='W0', w=32):
        """
        初始化PE（处理单元）
        :param name : 'W0', 'W1', 'W2'
        :param w: 特征图宽度，用于初始化Psum数组大小
        """
        # 权重寄存器，存储3个权重
        self.weights = torch.zeros(3, dtype=torch.float32)
        # Psum数组，长度为w，存储w个权重累加结果，位宽为16
        self.psum = torch.zeros(w, dtype=torch.float32)
        self.name = name
        # 记录当前r值，用于检测r切换
        self.current_r = -1
        # 记录当前c值，用于检测c切换
        self.current_c = None
        # PU索引
        self.pu_idx = None
        # 输出存储
        self.output = None
        # 暂存队列，存储(write_row, psum)元组
        self.pending_writes = []
    
    def set_weights(self, weights):
        """
        设置权重寄存器
        :param weights: 长度为3的权重列表或tensor
        """
        if len(weights) == 3:
            self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def set_output(self, output, pu_idx):
        """
        设置输出存储
        :param output: 输出tensor
        :param pu_idx: PU索引
        """
        self.output = output
        self.pu_idx = pu_idx
        # 根据output的宽度初始化psum
        if output is not None and output.shape[1] > 0:
            self.psum = torch.zeros(output.shape[2], dtype=torch.float32)
    
    def process(self, mask, c, r, H):
        """
        处理输入的values和c
        :param mask: 3个元素的列表，表示当前PE对应的3个特征图值
        :param c: 输出位置
        :param r: 当前行索引
        :param H: 输入特征图高度

        返回pusm和写请求以及写入of的行号
        """
        # 门控逻辑
        if not self._gating_rule(r, H):
            return -1, torch.zeros_like(self.psum)
        
        # 调试信息
        # if c != -1:
        #     print(f"PE {self.name}: process called with mask={mask}, c={c}, r={r}")
        
        # 算完一行了，第一个-1，直接写回
        if c == -1:
            # 调试信息
            # print(f"PE {self.name}: processing c=-1, r={r}, psum={self.psum}")
            if self.name == 'W2':
                psum_copy = self.psum.clone()
                self.reset()
                # print(f"PE {self.name}: returning r-2={r-2}, psum={psum_copy}")
                return self.current_r - 2, psum_copy
            elif self.name == 'W1':
                psum_copy = self.psum.clone()
                self.reset()
                # print(f"PE {self.name}: returning r-1={r-1}, psum={psum_copy}")
                return self.current_r - 1, psum_copy
            elif self.name == 'W0':
                psum_copy = self.psum.clone()
                self.reset()
                # print(f"PE {self.name}: returning r={r}, psum={psum_copy}")
                return self.current_r, psum_copy
        elif c < len(self.psum):
            # 记录r值，方便写入output_tensors
            self.current_r = r
            if(self.name == 'W1'):
                 print(f'self.current_r = {self.current_r}')
            # 计算权重累加结果
            result = torch.sum(mask * self.weights)
            
            # 将结果累加到Psum[c]
            self.psum[c] += result
            
            # 调试信息
            # print(f"PE {self.name}: updated psum[{c}] = {self.psum[c]}")

            # 计算还没有完成，返回-1和空的Psum数组
            return -1, torch.zeros_like(self.psum)
        else:
            return -1, torch.zeros_like(self.psum)

    def _gating_rule(self, r, H):
        """
        PE门控规则
        :param r: 当前行索引
        :param H: 输入特征图高度
        :return: 是否工作
        """
        if self.name == 'W2':
            # self.current_r = -1
            return not (r == 0 or r == 1)
        elif self.name == 'W1':
            # self.current_r = -1
            return not (r == 0 or r == H-1)
        elif self.name == 'W0':
            # self.current_r = -1
            return not (r == H-1 or r == H-2)
        return True
    
    def reset(self):
        """
        重置PE状态
        """
        self.psum = torch.zeros_like(self.psum)
