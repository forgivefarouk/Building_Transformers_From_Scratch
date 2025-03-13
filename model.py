import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model).float())
    
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super().__init__()
        sefl.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,d_model,2).float() * -(math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
    
    
class LayerNorm(nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    
    
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1 = torch.nn.Kinear(d_model,d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_ff,d_model)
    
    def froward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.dropout = nn.Dropout(dropout)
        self.wq = nn.Linear(d_model,d_model)
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)
        
        self.wo = nn.Linear(d_model,d_model)
        
    def forward(self,q,k,v,mask=None):
        batch_size = q.size(0)
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
    
        q = q.view(batch,q_len,self.heads,self.d_k).transpose(1,2)
        k = k.view(batch,k_len,self.heads,self.d_k).transpose(1,2)
        v = v.view(batch,v_len,self.heads,self.d_k).transpose(1,2)
        
        atten_weights = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            atten_weights = atten_weights.masked_fill(mask == 0,-1e9)
            
        atten_weights = F.softmax(atten_weights,dim=-1)
        atten_weights = self.dropout(atten_weights)
        out = torch.matmul(atten_weights,v)
        out = out.transpose(1,2).contiguous().view(batch,-1,self.d_model)
        return self.wo(out)
            
            
class Residual(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self,self_attn : MultiHeadAttention,feedforward : FeedForward,dropout=0.1):
        super().__init__()
        self.residual = nn.ModuleList([Residual(dropout) for _ in range(2)])
        self.attention = self_attn
        self.feed_forward = feedforward
    
    def forward(self,x,mask):
        x = self.residual[0](x,lambda x: self.attention(x,x,x,mask))
        x = self.residual[1](x,lambda x: self.feed_forward(x))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self,block,n_layers):
        super().__init__()
        self.layers = n_layers
        self.norm = LayerNorm()
        
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
    
    
class DecoderBlock(nn.Moduel):
    def __init__(self,self_attn : MultiHeadAttention,src_attn : MultiHeadAttention,feedforward : FeedForward,dropout=0.1):
        super().__init__()
        self.residual = nn.ModuleList([Residual(dropout) for _ in range(3)])
        self.self_attention = self_attn
        self.src_attention = src_attn
        self.feed_forward = feedforward
        
    def forward(self,x,enc_out,src_mask,tgt_mask):
        x = self.residual[0](x,lambda x: self.self_attention(x,x,x,tgt_mask))
        x = self.residual[1](x,lambda x: self.src_attention(x,enc_out,enc_out,src_mask))
        x = self.residual[2](x,lambda x: self.feed_forward(x))
        return x
    
    
class Decoder(nn.Module):
    def __init__(self,block,n_layers):
        super().__init__()
        self.layers = n_layers
        self.norm = LayerNorm()
        
    def forward(self,x,enc_out,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,enc_out,src_mask,tgt_mask)
        return self.norm(x)
    
    
class Projection(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)
    
class Trasformer(nn.Module):
    def __init__(self,src_vocab,tgt_vocab,d_model=512,n_layers=6,heads=8,d_ff=2048,dropout=0.1):
        super().__init__()
        self.encoder = Encoder(
            [EncoderBlock(MultiHeadAttention(d_model,heads,dropout),FeedForward(d_model,d_ff,dropout)) for _ in range(n_layers)]
        )
        self.decoder = Decoder(
            [DecoderBlock(MultiHeadAttention(d_model,heads,dropout),MultiHeadAttention(d_model,heads,dropout),FeedForward(d_model,d_ff,dropout)) for _ in range(n_layers)]
        )
        self.src_embed = nn.Sequential(InputEmbeddings(d_model,src_vocab),PositionalEncoding(d_model,dropout))
        self.tgt_embed = nn.Sequential(InputEmbeddings(d_model,tgt_vocab),PositionalEncoding(d_model,dropout))
        self.proj = Projection(d_model,tgt_vocab)
        
    def forward(self,src,tgt,src_mask,tgt_mask):
        enc_out = self.encoder(self.src_embed(src),src_mask)
        return self.proj(self.decoder(self.tgt_embed(tgt),enc_out,src_mask,tgt_mask))
    
    