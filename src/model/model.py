if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from base import BaseModel

'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, ffn_type="default",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = get_ffn(d_model, ffn_type)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)   # (bs, w*h, dim)

        # ffn
        src = self.ffn(src, spatial_shapes, level_start_index)

        return src
        '''
        
class GenClean(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(GenClean, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        # layers1 = []
        # layers1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        # layers1.append(nn.ReLU(inplace=True))
        # for _ in range(num_of_layers-2):
        #     layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        #     layers1.append(nn.BatchNorm2d(features, eps=0.0001, momentum = 0.95))
        #     layers1.append(nn.ReLU(inplace=True))
        # layers1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        # self.genclean = nn.Sequential(*layers1)
        # for n in self.genclean:
        #     if isinstance(n, nn.Conv2d):
        #         nn.init.orthogonal_(n.weight)
        #         print('init weight')
        #         if n.bias is not None:
        #             nn.init.constant_(n.bias, 0)
        #     elif isinstance(n, nn.BatchNorm2d):
        #         nn.init.constant_(n.weight, 1)
        #         nn.init.constant_(n.bias, 0)
        #laysers=TransformerEncoderLayer()
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight)
               nn.init.constant(m.bias, 0)
                
                
                
#         layers1 = []
#         layers1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
#         layers1.append(nn.ReLU(inplace=True))
#         for _ in range(num_of_layers-2):
#             layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
#             layers1.append(nn.BatchNorm2d(features, eps=0.0001, momentum = 0.95))
#             layers1.append(nn.ReLU(inplace=True))
#         layers1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
#         self.dncnn = nn.Sequential(*layers1)
#         for n in self.dncnn:
#             if isinstance(n, nn.Conv2d):
#                 nn.init.orthogonal_(m.weight)
#                 print('init weight')
#                 if n.bias is not None:
#                     nn.init.constant_(n.bias, 0)
#             elif isinstance(n, nn.BatchNorm2d):
#                 nn.init.constant_(n.weight, 1)
#                 nn.init.constant_(n.bias, 0)
            
                
        """
        x为输入的噪声图像
        """
    def forward(self, x):
        clean = self.genclean(x)
        # noisy=self.dncnn(x)
        return clean


class GenNoise(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(GenNoise, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

                
                
                
        layers1 = []
        layers1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        layers1.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers1.append(nn.BatchNorm2d(features, eps=0.0001, momentum = 0.95))
            layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers1)
        for n in self.dncnn:
            if isinstance(n, nn.Conv2d):
                nn.init.orthogonal_(n.weight)
                print('init weight')
                if n.bias is not None:
                    nn.init.constant_(n.bias, 0)
            elif isinstance(n, nn.BatchNorm2d):
                nn.init.constant_(n.weight, 1)
                nn.init.constant_(n.bias, 0)
	       
    def forward(self, x, weights=None, test=False):
        noisy=self.dncnn(x)
        return noisy

 
class CVF_model(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        FSize = 64
        self.gen_noise = GenNoise()
        self.genclean = GenClean()
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x, weights=None, test=False):                   

        clean = self.genclean(x)
        noisy = self.gen_noise(x)               
        return noisy,clean


        


