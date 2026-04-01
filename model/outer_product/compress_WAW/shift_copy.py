import torch

class split:
    """
    稀疏卷积加速器 Split 单元
    功能：将输入特征图行转换为位流及输出坐标 (r, c)
    """
    def __init__(self, kernel_size=3):
        self.k = kernel_size

    def process(self, if_line, r: int):
        """
        处理一行输入数据
        :param if_line: 输入特征图行tensor
        :param r: 当前行索引
        :return: (bitstream, (r, c))
        """
        # 确保输入是tensor
        if not isinstance(if_line, torch.Tensor):
            if_line = torch.tensor(if_line, dtype=torch.int32)
        
        # 转换为一维tensor
        if_line = if_line.view(-1)
        W = len(if_line)

        # 1. 提取非零元素的原始索引 idx
        nz_indices = torch.nonzero(if_line, as_tuple=False).squeeze().tolist()
        if not isinstance(nz_indices, list):
            nz_indices = [nz_indices] if nz_indices is not None else []

        if len(nz_indices) == 0:
            return torch.tensor([], dtype=torch.int32), (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))
        
        # 2. 压缩bitstream
        bitstream_list = []
        zero_count = 0
        for i in range(W):
            if if_line[i] == 0:
                zero_count += 1
                if i == W-1:
                    if zero_count > 2:
                        bitstream_list.extend([0, 0])
                    else:
                        bitstream_list.extend([0] * zero_count)
            # 检测到1了
            else:
                if zero_count > 2:
                    bitstream_list.extend([0, 0, 1])
                else:
                    bitstream_list.extend(if_line[i-zero_count:i+1].tolist())
                zero_count = 0
        
        # 转换为tensor
        bitstream = torch.tensor(bitstream_list, dtype=torch.int32)

        # 3. 根据 idx 确定输出位置 c
        # 规则：对于每个 j, 贡献到窗口 k \in [j-2, j], 且 0 <= k <= W-3
        c_set = set()
        max_c = W - self.k
        for j in nz_indices:
            start_k = max(0, j - (self.k - 1))
            end_k = min(j, max_c)
            for k in range(start_k, end_k + 1):
                c_set.add(k)
        
        # 4. 转换为tensor输出c
        c_array = torch.tensor(sorted(list(c_set)), dtype=torch.int32)
        r_array = torch.tensor([r]*len(c_array), dtype=torch.int32)

        return bitstream, (r_array,c_array)


class SplitUnit:
    """
    接收一整个输入特征图[H,W]
    输出压缩后的bitstream，r，c
    """
    def __init__(self, kernel_size=3):
        self.k = kernel_size
        self.split = split(kernel_size=kernel_size)

    def process(self, if_map):
        bitstream_list = []
        r_list = []
        c_list = []
        for r in range(if_map.shape[0]):
            bitstream, (r_array, c_array) = self.split.process(if_map[r], r)
            # 添加predictor，如果发现某个bitstream的长度为3/6，则根据现有bitstream_list中的len，预测几拍后发生WAW，但是不影响list的生成
            if bitstream.shape[0] == 3 or bitstream.shape[0] == 6:
                total_length = sum(bs.shape[0] for bs in bitstream_list)
                if bitstream.shape[0] == 3:
                    print(f"{total_length}拍后,第{r}行的bitstream长度为{bitstream.shape[0]}，发生WAW的可能cycle为{total_length+bitstream.shape[0]}和{total_length+bitstream.shape[0]+3}")
                else:
                    print(f"{total_length}拍后,第{r}行的bitstream长度为{bitstream.shape[0]}，发生WAW的可能cycle为{total_length+bitstream.shape[0]}")

            bitstream_list.append(bitstream)
            # 添加空的两拍 -1,-1
            r_with_sep = torch.cat([r_array, torch.tensor([-1, -1], dtype=r_array.dtype, device=r_array.device)])
            c_with_sep = torch.cat([c_array, torch.tensor([-1, -1], dtype=c_array.dtype, device=c_array.device)])
            r_list.append(r_with_sep)
            c_list.append(c_with_sep)
        # 拼接所有行的结果
        bitstream_all = torch.cat(bitstream_list) if bitstream_list else torch.tensor([], dtype=torch.int32)
        r_all = torch.cat(r_list) if r_list else torch.tensor([], dtype=torch.int32)
        c_all = torch.cat(c_list) if c_list else torch.tensor([], dtype=torch.int32)
        return bitstream_all, r_all, c_all            

    # def process(self, if_map):
        bitstream_list = []
        r_list = []
        c_list = []
        for r in range(if_map.shape[0]):
            bitstream, (r_array, c_array) = self.split.process(if_map[r], r)
            # 添加predictor，如果发现某个bitstream的长度为3/6，则根据现有bitstream_list中的len，预测几拍后发生WAW，并插入一个nop
            if bitstream.shape[0] == 3 or bitstream.shape[0] == 6:
                total_length = sum(bs.shape[0] for bs in bitstream_list)
                print(f"{total_length+3}拍后,第{r}行的bitstream长度为{bitstream.shape[0]}，发生WAW的可能cycle为{total_length+3+bitstream.shape[0]}")
                print(f"我插入nop后，实际运行结果：")
                # 添加空的三拍 -1,-1,-1，bitstream也多一个0
                bitstream_list.append(torch.cat([bitstream, torch.tensor([0], dtype=bitstream.dtype, device=bitstream.device)]))
                r_with_sep = torch.cat([r_array, torch.tensor([-1, -1, -1], dtype=r_array.dtype, device=r_array.device)])
                c_with_sep = torch.cat([c_array, torch.tensor([-1, -1, -1], dtype=c_array.dtype, device=c_array.device)])
                r_list.append(r_with_sep)
                c_list.append(c_with_sep)
            else:
                bitstream_list.append(bitstream)
                # 添加空的两拍 -1,-1
                r_with_sep = torch.cat([r_array, torch.tensor([-1, -1], dtype=r_array.dtype, device=r_array.device)])
                c_with_sep = torch.cat([c_array, torch.tensor([-1, -1], dtype=c_array.dtype, device=c_array.device)])
                r_list.append(r_with_sep)
                c_list.append(c_with_sep)
        # 拼接所有行的结果
        bitstream_all = torch.cat(bitstream_list) if bitstream_list else torch.tensor([], dtype=torch.int32)
        r_all = torch.cat(r_list) if r_list else torch.tensor([], dtype=torch.int32)
        c_all = torch.cat(c_list) if c_list else torch.tensor([], dtype=torch.int32)
        return bitstream_all, r_all, c_all            
# 