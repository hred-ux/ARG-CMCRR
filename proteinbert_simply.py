import math
import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange

class MultiHeadAttentionWithRelativePos(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8, max_rel_pos=50):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_rel_pos = max_rel_pos

        self.rel_pos_emb = nn.Embedding(2*max_rel_pos+1, self.head_dim)

        self.qkv = nn.Linear(hidden_dim, 3*hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def _get_rel_pos(self, seq_len, device):

        range_vec = torch.arange(seq_len, device=device)
        rel_pos = range_vec[None, :] - range_vec[:, None]
        rel_pos = torch.clamp(rel_pos + self.max_rel_pos, 0, 2*self.max_rel_pos)
        return rel_pos

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        rel_pos = self._get_rel_pos(L, x.device)
        rel_emb = self.rel_pos_emb(rel_pos)
        rel_emb = rel_emb.unsqueeze(0).repeat(B,1,1,1)

        attn_score = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        rel_score = torch.einsum('bhld,blrd->bhlr', q, rel_emb) / math.sqrt(self.head_dim)
        attn_score += rel_score
        
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn_score, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1,2).reshape(B, L, -1)
        return self.proj(context)

class ProteinBERT(nn.Module):
    def __init__(
        self,
        dim=128,
        depth=2,
        narrow_conv_kernel=9,
        wide_conv_kernel=9,
        wide_conv_dilation=5,
        attn_heads=8
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, narrow_conv_kernel, padding=narrow_conv_kernel // 2),
                nn.GELU(),
                nn.Conv1d(dim, dim, wide_conv_kernel, dilation=wide_conv_dilation, 
                          padding=(wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2),
                nn.GELU(),
                Rearrange('b c s -> b s c'),

                MultiHeadAttentionWithRelativePos(hidden_dim=dim, num_heads=attn_heads, max_rel_pos=50),
                nn.LayerNorm(dim),

                Rearrange('b s c -> b c s')
            ) for _ in range(depth)
        ])

        self.to_output = nn.Sequential(
            Rearrange('b c s -> b s c'),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)

        return self.to_output(x)