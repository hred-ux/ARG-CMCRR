from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from models.proteinbert_simply import ProteinBERT


class GatedFusion(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim, dropout=0.3):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.a_proj = nn.Sequential(nn.Linear(dim_a, out_dim), nn.GELU())
        self.b_proj = nn.Sequential(nn.Linear(dim_b, out_dim), nn.GELU())
        self.gate = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        if a.size(-1) != self.dim_a or b.size(-1) != self.dim_b:
            raise ValueError(
                "GatedFusion input dimensions do not match "
                f"dim_a={self.dim_a}, dim_b={self.dim_b}: "
                f"got {a.size(-1)} and {b.size(-1)}"
            )
        a = self.a_proj(a)
        b = self.b_proj(b)
        gate = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        fused = gate * a + (1.0 - gate) * b
        return self.dropout(self.norm(fused))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return self.proj(context)

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

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size=22, embedding_dim=128, hidden_dim=128, num_classes=10,
                 dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.embed_dropout = nn.Dropout(0.2)
        
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, hidden_dim // 2, k, padding=k//2),
                nn.BatchNorm1d(hidden_dim//2),
                nn.GELU(),
                nn.Dropout(0.3)
            ) for k in [3, 5]
        ])
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.attention = MultiHeadSelfAttention(
            hidden_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout
        )
        
        self.protein_bert = ProteinBERT(
            num_tokens=vocab_size,
            dim=embedding_dim,
            depth=2,
            attn_heads=4,
            attn_dim_head=64
        )
        self.bert_adapter = nn.Linear(embedding_dim, hidden_dim*2)
        
        
    
        self.fusion = GatedFusion(
            dim_a=hidden_dim * 2,
            dim_b=hidden_dim * 2,
            out_dim=hidden_dim * 2,
            dropout=0.3
        )
    
        self.pool = nn.AdaptiveAvgPool1d(1)
        
      
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, x_len):
        
        x = self.embedding(x)  
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)
        
        
        conv_out = x.transpose(1, 2)  
        conv_features = [conv(conv_out) for conv in self.conv_blocks]
        conv_features = torch.cat(conv_features, dim=1)
        conv_features = conv_features.transpose(1, 2)
        
       
        packed = pack_padded_sequence(conv_features, x_len, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        
   
        attn_out = self.attention(gru_out) 
        
    
        bert_feat = self.protein_bert(x)  
        bert_feat = self.bert_adapter(bert_feat)
        
        current_max_len = attn_out.size(1)
        bert_feat = bert_feat[:, :current_max_len, :] 
        
        fused = self.fusion(attn_out, bert_feat)
        
        global_max_pool = torch.max(fused, dim=1)[0]
        global_avg_pool = torch.mean(fused, dim=1)
        final_features = (global_max_pool + global_avg_pool) / 2
        
        return self.classifier(final_features)









        














    


    
        
        
        
        
        

        

        
        
        
        

        
        

        

        

