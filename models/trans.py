import torch
from torch import nn
from torch.nn import functional as F



class MTv00(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.init_cnn = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1)
        
        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        
        # building encoder
        for ind, ((dim_in_x, dim_out_x), (dim_in_g, dim_out_g)) in enumerate(zip(in_out_x, in_out_g)):
            is_last = ind >= (num_resolutions - 1)
            
            encoder = EM(dim_x=dim_in_x, dim_g=dim_in_g, time_x=time_dim_x, time_g=time_dim_g, resnet_block_groups=resnet_block_groups)
            g_down = nn.Conv2d(dim_in_g+dim_in_x, dim_out_g, 3, padding=1) if is_last else Downsample(dim_in_g+dim_in_x, dim_out_g)
            x_down = nn.Conv2d(dim_in_x, dim_out_x, 3, padding=1) if is_last else Downsample(dim_in_x, dim_out_x)
            self.encoder_blocks.append(nn.ModuleList([encoder, g_down, x_down]))
    
            
        
    def forward(self, x):
        x = self.init_cnn(x)
        return x.sum()
        
        