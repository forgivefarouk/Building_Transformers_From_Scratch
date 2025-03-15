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
    def __init__(self,d_model,max_len=5000,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
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
        self.linear1 = torch.nn.Linear(d_model,d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_ff,d_model)
    
    def forward(self,x):
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
        batch = q.size(0)
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
    def __init__(self,features , dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self,features,self_attn : MultiHeadAttention,feedforward : FeedForward,dropout=0.1):
        super().__init__()
        self.residual = nn.ModuleList([Residual(features , dropout) for _ in range(2)])
        self.attention = self_attn
        self.feed_forward = feedforward
    
    def forward(self,x,mask):
        x = self.residual[0](x,lambda x: self.attention(x,x,x,mask))
        x = self.residual[1](x,lambda x: self.feed_forward(x))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self,features,n_layers):
        super().__init__()
        self.layers = n_layers
        self.norm = LayerNorm(features)
        
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
    
    
class DecoderBlock(nn.Module):
    def __init__(self,features,self_attn : MultiHeadAttention,src_attn : MultiHeadAttention,feedforward : FeedForward,dropout=0.1):
        super().__init__()
        self.residual = nn.ModuleList([Residual(features,dropout) for _ in range(3)])
        self.self_attention = self_attn
        self.src_attention = src_attn
        self.feed_forward = feedforward
        
    def forward(self,x,enc_out,src_mask,tgt_mask):
        x = self.residual[0](x,lambda x: self.self_attention(x,x,x,tgt_mask))
        x = self.residual[1](x,lambda x: self.src_attention(x,enc_out,enc_out,src_mask))
        x = self.residual[2](x,lambda x: self.feed_forward(x))
        return x
    
    
class Decoder(nn.Module):
    def __init__(self,features ,n_layers):
        super().__init__()
        self.layers = n_layers
        self.norm = LayerNorm(features)
        
    def forward(self,x,enc_out,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,enc_out,src_mask,tgt_mask)
        return self.norm(x)
    
    
class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer