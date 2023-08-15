import torch
from torch import nn
from functools import partial
import einops
import numpy as np


__all__ = ["TSegDiff",]



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def embed(self, p, dim):
        half_dim = dim // 2
        embeddings = torch.log(torch.tensor(self.dim//16)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=p.device) * -embeddings)
        embeddings = p[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def forward(self, p_r, p_c):
        half_dim = self.dim // 2
        embeddings_r = self.embed(p_r, half_dim)
        embeddings_c = self.embed(p_c, half_dim)
        embeddings = torch.cat((embeddings_r, embeddings_c), dim=-1)
        return embeddings
    


class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        patch_size: Tuple[int, int],
        latent_size: int,
        hw_size: Tuple[int, int],
        n_channel: int,
        batch_size: int,
        device: torch.device.type,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.patch_size = patch_size
        
        self.linear_projection = nn.Linear(patch_size[0]*patch_size[1]*n_channel, latent_size)
        self.segment_token = nn.Parameter(torch.randn(batch_size, 1, latent_size), requires_grad=True).to(device)
        
        p_r = torch.arange(1, hw_size[0]//patch_size[0]+1)
        p_c = torch.arange(1, hw_size[1]//patch_size[1]+1)
        poses = torch.cartesian_prod(p_r, p_c)+1
        poses = torch.concat([torch.zeros((1, 2)), poses], dim=0)
        sembed = SinusoidalPositionEmbeddings(dim=latent_size)
        self.pos_embedding = sembed(poses[:,0], poses[:,1]).to(device)
        
    def forward(self, x):
        patches = einops.rearrange(
            x, 
            "b c (h h1) (w w1) -> b (h w) (h1 w1 c)", 
            h1=self.patch_size[0], w1=self.patch_size[1]
        )
        projected_patches = self.linear_projection(patches)
        x = torch.concat([self.segment_token, projected_patches], dim=1)
        pos_embedding = einops.repeat(self.pos_embedding, "p d -> b p d", b=x.shape[0])
        
        print(x.shape, pos_embedding.shape)
        x += pos_embedding
        return x



class MLP(nn.Module):
    def __init__(self, dim: int, dropout=0.5, hidden_scale=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*hidden_scale),
            nn.GELU(),
            nn.Linear(dim*hidden_scale, dim),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        return self.mlp(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,  
        num_head,
        embed_dim,
        dropout = 0.5,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.mlpn = MLP(embed_dim, dropout=dropout, hidden_scale=4)
        self.norm = nn.LayerNorm()
        self.heads = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=dropout)
        
    def forward(self, x):
        x += self.heads(self.norm(x))
        x += self.mlp(self.norm(x))[0]
        return x



class TSegDiff(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        
        **kwargs
    ):
        '''
        `x` and `g` refer to target and guidance correspondingly.
        '''
        super().__init__()
        
        self.init_filter=32
        self.patch_size = 32
        
        self.init_conv = nn.Conv2d(in_ch, self.init_filter, kernel_size=(3, 3))
        self.patch_embedding = PatchEmbedding(self.patch_size, )
        
    def forward(self, x):
        
        x = self.init_conv(x)
        # [b, 32, h, w]
        
        x = image_to_patches(x, p=self.patch_size)
        
        x = self.linear_1(x)
        # [b, (hw/p^2), p^2c] -> [b, (hw/p^2), d]
        
        
        
        

    