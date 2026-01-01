import torch
import torch.nn as nn
import math
from config import GPTConfig


class CasualMultiHeadAttention(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.n_embd=config.n_embd
        self.n_heads=config.n_heads
        self.head_dim=config.n_embd//config.n_heads
        self.W_Query=nn.Linear(self.n_embd,self.n_embd) #Here emb_in and emb_out are same i.e. 256
        self.W_Key=nn.Linear(self.n_embd,self.n_embd)
        self.W_Value=nn.Linear(self.n_embd,self.n_embd)
        self.proj=nn.Linear(self.n_embd,self.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask",torch.tril(torch.ones(config.block_size,config.block_size)))
    

    def forward(self,x,past_kv=None):
        batch_size,no_of_tokens,input_dimension=x.shape #(B,T,C)

        keys=self.W_Key(x) #multiplication of X(batch_size x no_of_tokens x emb_in) with W_Q(emb_in x emb_out) 
                           # i.e shape : (batch_size,no_of_tokens,emb_out)
        queries=self.W_Query(x)
        values=self.W_Value(x)

        #emb_out = no_of_heads * head_dimension  (as we know)

        keys=keys.view(batch_size,no_of_tokens,self.n_heads,self.head_dim) # shape : (batch_size , no_of_tokens , no_of_heads , head_dimension)
        queries=queries.view(batch_size,no_of_tokens,self.n_heads,self.head_dim)
        values=values.view(batch_size,no_of_tokens,self.n_heads,self.head_dim)

        keys=keys.transpose(1,2) # shape : (batch_size , no_of_heads , no_of_tokens , head_dimension)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)

        if past_kv is not None:
            past_k, past_v = past_kv
            keys = torch.cat([past_k, keys], dim=2)
            values = torch.cat([past_v, values], dim=2)

        attn_scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)#dot product -> shape:(batch_size , no_of_heads , head_dimension , head_dimension)
        # attn_scores = attn_scores.masked_fill(self.mask[:no_of_tokens, :no_of_tokens] == 0, float("-inf"))
        
        Tq = queries.size(2)
        Tk = keys.size(2)   

        attn_scores = attn_scores.masked_fill(
            self.mask[:Tq, :Tk] == 0,
            float("-inf")
        )
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        out = attn_scores @ values
        out = out.transpose(1, 2).contiguous().view(batch_size,no_of_tokens,input_dimension)
        return self.proj(out),(keys,values)