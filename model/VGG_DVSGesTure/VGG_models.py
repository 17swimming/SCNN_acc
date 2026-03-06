# import random
# from models.layers import *



# class VGGSNN(nn.Module):
#     def __init__(self):
#         super(VGGSNN, self).__init__()
#         pool = SeqToANNContainer(nn.AvgPool2d(2))
#         #pool = APLayer(2)
#         self.features = nn.Sequential(
#             Layer(2,64,3,1,1),
#             Layer(64,128,3,1,1),
#             pool,
#             Layer(128,256,3,1,1),
#             Layer(256,256,3,1,1),
#             pool,
#             Layer(256,512,3,1,1),
#             Layer(512,512,3,1,1),
#             pool,
#             Layer(512,512,3,1,1),
#             Layer(512,512,3,1,1),
#             pool,
#         )
#         W = int(48/2/2/2/2)
#         # self.T = 4
#         self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, input):
#         # input = add_dimention(input, self.T)
#         x = self.features(input)
#         x = torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x

# class VGGSNNwoAP(nn.Module):
#     def __init__(self):
#         super(VGGSNNwoAP, self).__init__()
#         self.features = nn.Sequential(
#             Layer(2,64,3,1,1),
#             Layer(64,128,3,2,1),
#             Layer(128,256,3,1,1),
#             Layer(256,256,3,2,1),
#             Layer(256,512,3,1,1),
#             Layer(512,512,3,2,1),
#             Layer(512,512,3,1,1),
#             Layer(512,512,3,2,1),
#         )
#         W = int(48/2/2/2/2)
#         # self.T = 4
#         self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, input):
#         # input = add_dimention(input, self.T)
#         x = self.features(input)
#         x = torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x



# if __name__ == '__main__':
#     model = VGGSNNwoAP()


# ------------------------------我的修改版------------------------------------
import torch
import torch.nn as nn
from layers import *

class VGGSNN(nn.Module):
    # 增加 num_classes 参数（默认11适应Gesture），增加 img_width (默认48适应论文设置)
    def __init__(self, num_classes=11, img_width=48):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        
        # 输入通道固定为 2 (DVS数据)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            Layer(64, 128, 3, 1, 1),
            pool, # 48 -> 24
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            pool, # 24 -> 12
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool, # 12 -> 6
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool, # 6 -> 3
        )
        
        # 动态计算全连接层的输入维度
        # 经过4次池化(每次除以2)，最终宽度是原始宽度除以16
        W = int(img_width / 16) 
        
        # self.T = 4 # T 会在 forward 或者外部设置
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input: [N, T, 2, H, W]
        # 不需要 add_dimention，因为 DVS 数据自带 T 维度
        # input = add_dimention(input, self.T) 
        
        x = self.features(input)
        x = torch.flatten(x, 2) # [N, T, 512*W*W]
        x = self.classifier(x)  # [N, T, num_classes]
        return x