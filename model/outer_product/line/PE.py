import torch

class PE:
    def __init__(self, w=32):
        """
        初始化PE（处理单元）
        :param w: 特征图宽度，用于初始化Psum数组大小
        """
        # 权重寄存器，存储3个权重
        self.weights = torch.zeros(3, dtype=torch.float32)
        # Psum数组，长度为w，存储w个权重累加结果，位宽为16
        self.psum = torch.zeros(w, dtype=torch.float32)
    
    def set_weights(self, weights):
        """
        设置权重寄存器
        :param weights: 长度为3的权重列表或tensor
        """
        if len(weights) == 3:
            self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def process(self, mask, c):
        """
        处理输入的mask和c
        :param mask: 3bit的掩码
        :param c: 输出位置
        """
        if c == -1:
            return
        elif c < len(self.psum):
            # 将mask转换为tensor
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            # 计算权重累加结果
            result = torch.sum(mask_tensor * self.weights)
            
            # 将结果存放到Psum[c]
            self.psum[c] = result
    
    def get_psum(self,w):
        """
        获取Psum数组的前w个元素
        """
        return self.psum[:w]
    
    def reset(self):
        """
        重置PE状态
        """
        self.psum = torch.zeros_like(self.psum)
